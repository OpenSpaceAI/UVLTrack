import torch
import torch.nn.functional as F
from torch import nn

from .mae_vit import mae_vit_base_patch16, mae_vit_large_patch16

from .bert_backbone import BertModel
import numpy as np


class ModalityUnifiedFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        """ Initializes the model."""
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_layer = cfg.MODEL.BACKBONE.FUSION_LAYER
        self.cont_loss_layer = cfg.MODEL.BACKBONE.CONT_LOSS_LAYER
        self.txt_token_mode = cfg.MODEL.BACKBONE.TXT_TOKEN_MODE
        
        if 'base' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_base_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE), 
                                            learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'], strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            self.bert.encoder.layer = self.bert.encoder.layer[:min(self.fusion_layer)]
            
        elif 'large' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_large_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE), 
                                            learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'], strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            self.bert.encoder.layer = self.bert.encoder.layer[:min(self.fusion_layer)]
            
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

    def cat_mask(self, text, flag):
        x_mask = torch.ones([flag.shape[0], self.vit.num_patches_x]).to(flag.device)
        z_mask = torch.ones([flag.shape[0], self.vit.num_patches_z]).to(flag.device)*(flag!=1) # =1 mask
        c_mask = torch.ones([flag.shape[0], 1]).to(flag.device)*(flag!=1) # =1 mask
        t_mask = text.mask*(flag!=0) # =0 mask
        mask = ~torch.cat([c_mask, z_mask, x_mask, t_mask], dim=1).bool()
        visual_mask = ~torch.cat([c_mask, z_mask, x_mask], dim=1).bool()
        return mask, visual_mask

    def forward(self, template, search, text, flag): # one more token
        img_feat = self.vit.patchify(template, search)
        txt_feat, bert_mask = self.bert.embedding(text.tensors, token_type_ids=None, attention_mask=text.mask)
        mask, visual_mask = self.cat_mask(text, flag)
        logits_list = []
        for i in range(len(self.vit.blocks)):
            if i in self.fusion_layer:
                img_feat, txt_feat = self.vit.forward_joint(img_feat, txt_feat, mask, i, flag=flag)
            else:
                img_feat = self.vit.blocks[i](img_feat, visual_mask, flag=flag)
                txt_feat = self.bert.encoder.layer[i](txt_feat, bert_mask)
            if i in self.cont_loss_layer:
                logits = self.contractive_learning(img_feat, txt_feat, text, flag)
                logits_list.append(logits)
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)
        b, s, c = x.shape
        out_dict = {
            "search": x,
            "template": z,
            "text": txt_feat,
            "vis_token": vis_token,
            "txt_token": self.generate_txt_token(txt_feat, text),
            "flag": flag.reshape(-1),
            "logits": torch.stack(logits_list, dim=1).reshape(b, -1, int(s**0.5), int(s**0.5))
        }
        return out_dict
    
    def generate_txt_token(self, txt_feat, text):
        if self.txt_token_mode == 'mean':
            return (txt_feat*text.mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / text.mask.unsqueeze(-1).sum(dim=1, keepdim=True)
        elif self.txt_token_mode == 'cls':
            return txt_feat[:, :1]
    
    def contractive_learning(self, img_feat, txt_feat, text, flag):
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)
        txt_token = self.generate_txt_token(txt_feat, text)
        vis_logits = self.logit_scale.exp()*(F.normalize(x, dim=-1) @ F.normalize(vis_token, dim=-1).transpose(-2,-1))
        txt_logits = self.logit_scale.exp()*(F.normalize(x, dim=-1) @ F.normalize(txt_token, dim=-1).transpose(-2,-1))
        logits_group = torch.stack([vis_logits, txt_logits, (vis_logits+txt_logits)/2], dim=1)
        bid = torch.arange(flag.shape[0])
        logits = logits_group[bid, flag.reshape(-1)]
        return logits
        
        

def modality_unified_feature_extractor(cfg):
    model = ModalityUnifiedFeatureExtractor(cfg)
    return model
