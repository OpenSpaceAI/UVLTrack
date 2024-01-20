from lib import registry
from .modality_unified_feature_extractor import modality_unified_feature_extractor

@registry.BACKBONES.register('modality_unified_feature_extractor')
def build_modality_unified_feature_extractor(cfg):
    vit = modality_unified_feature_extractor(cfg)
    return vit