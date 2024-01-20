import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import numpy as np

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class DistributionBasedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_rate=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 / 50

        self.q  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop_tgt = nn.Dropout(drop_rate)
        self.attn_drop_dis = nn.Dropout(drop_rate)
        self.attn_drop_bgd = nn.Dropout(drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_embed = nn.Embedding(3, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def divide_background(self, bgd_score):
        # probability density function
        values, indices = bgd_score.sort(dim=-1, descending=False)
        mask = values.cumsum(dim=-1) < 0.25
        
        # divide background features into background and distractors
        threshold = values.masked_fill(mask, 1.0).min(dim=-1, keepdim=True).values # set background to 1 and obtain the min score of distractors
        
        # distractor mask
        mask = bgd_score >= threshold
        return mask
    
    def distribute_attn(self, tgt, sim_logit, tgt_mask):
        # target token
        tgt_score = sim_logit.masked_fill(~tgt_mask, -1e20).softmax(-1)
        tgt_token = self.attn_drop_tgt(tgt_score) @ tgt
        
        # background score
        bgd_logit = sim_logit.masked_fill( tgt_mask, -1e20)
        bgd_score = bgd_logit.softmax(-1)
        
        # divide backgrounds into backgrounds and distractors
        dis_mask = self.divide_background(bgd_score)
        
        # background token and distractor token
        bgd_score = bgd_logit.masked_fill( dis_mask, -1e20).softmax(-1) # pure background scores
        dis_score = bgd_logit.masked_fill(~dis_mask, -1e20).softmax(-1) # distractor scores
        
        bgd_token = self.attn_drop_bgd(bgd_score) @ tgt
        dis_token = self.attn_drop_dis(dis_score) @ tgt
        return tgt_token, bgd_token, dis_token
        

    def forward(self, tem, tem_mask, ctx, ctx_mask, cls_token, flag): # TODO: grounding 不需要更新
        src_ = self.query_embed.weight.unsqueeze(0).repeat(ctx.shape[0], 1, 1) # B, 2, C
        src_[:, 0] = src_[:, 0] + cls_token
        
        # concatenate context feature and template feature
        tgt = torch.cat([tem, ctx], dim=1)
        tgt_mask = torch.cat([tem_mask, ctx_mask], dim=1).unsqueeze(1)
        
        # the similarity between clstoken and context features
        sim_logit = F.normalize(cls_token, dim=-1).unsqueeze(1) @ F.normalize(tgt, dim=-1).transpose(-2,-1) * self.logit_scale.exp()
        
        # divide context features based on distribution
        tgt_token, bgd_token, dis_token = self.distribute_attn(tgt, sim_logit, tgt_mask)
        src = torch.cat([tgt_token, dis_token, bgd_token], dim=1) + src_
        src = self.mlp(src) + src
        
        # switcher
        src = torch.stack([src, src_, src], dim=1)
        bid = torch.arange(tgt.shape[0]).cuda()
        src = src[bid, flag]
        
        return src
        
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))