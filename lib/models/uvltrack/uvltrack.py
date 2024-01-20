from torch import nn

from lib import registry
from lib.models.backbones import *
from lib.models.heads import *


class UVLTrack(nn.Module):
    """ This is the base class for Transformer Tracking, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head):
        """ Initializes the model.
        """
        super().__init__()
        # self.language_backbone = language_backbone
        self.backbone = backbone
        self.box_head = box_head

    def forward(self, template, search, text, template_mask, context_mask, flag):
        # text_feature = self.language_backbone(text) # b, s, c  b, s  FT
        backbone_info = self.backbone(template, search, text, flag)
        backbone_info['template_mask'] = template_mask
        backbone_info['context_mask'] = context_mask
        head_info = self.box_head(backbone_info)
        return head_info
    
    def forward_prompt_init(self, template, search, text, template_mask, context_mask, flag):
        backbone_info = self.backbone(template, search, text, flag)
        backbone_info['template_mask'] = template_mask
        backbone_info['context_mask'] = context_mask
        prompt = self.box_head.forward_prompt(backbone_info)
        return prompt
    
    def forward_prompt(self, out_dict, template_mask, context_mask):
        backbone_info = out_dict
        backbone_info['template_mask'] = template_mask
        backbone_info['context_mask'] = context_mask
        prompt = self.box_head.forward_prompt(backbone_info)
        return prompt
        
    
    def forward_test(self, template, search, text, prompt, flag):
        backbone_info = self.backbone(template, search, text, flag)
        backbone_info['prompt'] = prompt
        head_info = self.box_head(backbone_info)
        return head_info
        
@registry.MODELS.register('uvltrack')
def build_model(cfg):
    # language_backbone = registry.BACKBONES[cfg.MODEL.BACKBONE.LANGUAGE.TYPE](cfg)
    backbone = registry.BACKBONES[cfg.MODEL.BACKBONE.TYPE](cfg)
    head = registry.HEADS[cfg.MODEL.HEAD.TYPE](cfg)  # a simple corner head
    model = UVLTrack(
        # language_backbone,
        backbone,
        head
    )
    return model
