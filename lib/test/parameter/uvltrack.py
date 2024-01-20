from lib.test.utils import TrackerParams
import os
from easydict import EasyDict as edict
from lib.test.evaluation.environment import env_settings
from lib.config.uvltrack.config import cfg, update_config_from_file


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, edict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return

def parameters(yaml_name: str, extra_cfg=None, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    params.yaml_name = yaml_name
    yaml_file = os.path.join(prj_dir, 'experiments/uvltrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    if epoch is not None:
        cfg.TEST.EPOCH = epoch
    params.cfg = cfg
    if extra_cfg is not None:
        _update_config(params.cfg, extra_cfg)
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.grounding_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/uvltrack/%s/UVLTrack_ep%04d.pth.tar"%(yaml_name, cfg.TEST.EPOCH))  # 470

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
