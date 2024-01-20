from lib import registry
from .modality_adaptive_box_head import ModalityAdaptiveBoxHead

@registry.HEADS.register('modality_adaptive_box_head')
def build_modality_adaptive_box_head(cfg):
    stride = 16
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = cfg.MODEL.HEAD.HEAD_DIM
    head = ModalityAdaptiveBoxHead(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel, feat_sz=feat_sz, stride=stride, 
                        cls_tokenize=cfg.MODEL.HEAD.CLS_TOKENIZE, offset_sigmoid=cfg.MODEL.HEAD.OFFSET_SIGMOID,
                        joint_cls=cfg.MODEL.HEAD.JOINT_CLS, drop_rate=cfg.MODEL.HEAD.DROP, softmax_one=cfg.MODEL.HEAD.SOFTMAX_ONE,
                        grounding_dilation=cfg.MODEL.HEAD.GROUNDING_DILATION, contrastive_conv=cfg.MODEL.HEAD.CONTRASTIVE_CONV)
    return head