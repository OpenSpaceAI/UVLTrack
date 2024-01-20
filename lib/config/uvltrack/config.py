from easydict import EasyDict as edict
import yaml

"""
Add default config for MixFormer.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HIDDEN_DIM = 384
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # sine or learned
cfg.MODEL.PREDICT_MASK = False
cfg.MODEL.LEARNABLE_POSITION = False
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = 'mae_vit'
cfg.MODEL.BACKBONE.DROP_PATH_RATE = 0.0
cfg.MODEL.BACKBONE.PRETRAINED_PATH = ''
cfg.MODEL.BACKBONE.FUSION_LAYER = [8,9,10,11]
cfg.MODEL.BACKBONE.CONT_LOSS_LAYER = [4,5,6,7,8,9,10,11]
cfg.MODEL.BACKBONE.TXT_TOKEN_MODE = 'token'

cfg.MODEL.BACKBONE.LANGUAGE = edict()
cfg.MODEL.BACKBONE.LANGUAGE.IMPLEMENT = 'pytorch'
cfg.MODEL.BACKBONE.LANGUAGE.TYPE = 'bert-base-uncased'
cfg.MODEL.BACKBONE.LANGUAGE.PATH = 'pretrained/bert/bert-base-uncased.tar.gz'
cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH = 'pretrained/bert/bert-base-uncased-vocab.txt'
# BERT
cfg.MODEL.BACKBONE.LANGUAGE.BERT = edict()
cfg.MODEL.BACKBONE.LANGUAGE.BERT.LR = 10e-5
cfg.MODEL.BACKBONE.LANGUAGE.BERT.ENC_NUM = 12
cfg.MODEL.BACKBONE.LANGUAGE.BERT.HIDDEN_DIM = 256
cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN = 40

cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = 'anchor_free'
cfg.MODEL.HEAD.HEAD_DIM = 384
cfg.MODEL.HEAD.CLS_TOKENIZE = True
cfg.MODEL.HEAD.OFFSET_SIGMOID = True
cfg.MODEL.HEAD.JOINT_CLS = False
cfg.MODEL.HEAD.DROP = 0.0
cfg.MODEL.HEAD.SOFTMAX_ONE = False
cfg.MODEL.HEAD.GROUNDING_DILATION = 1
cfg.MODEL.HEAD.CONTRASTIVE_CONV = False


# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.POSITIVE_MODE = 'ctr'
cfg.TRAIN.MODE = 'grounding'
cfg.TRAIN.VLTVG_AUG = False
cfg.TRAIN.GROUNDING_RATIO = None
cfg.TRAIN.VL_RATIO = None
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.AUX_WEIGHT = 0.0
cfg.TRAIN.CONT_WEIGHT = 1.0
cfg.TRAIN.CIB_WEIGHT = 0.01
cfg.TRAIN.CTR_RATIO = 0.75
cfg.TRAIN.DEEP_SUPERVISION = False
cfg.TRAIN.FREEZE_STAGE0 = False
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.DYNAMIC_CLS = False
cfg.TRAIN.REDUCTION = 'sum'
cfg.TRAIN.GAUSSIAN_IOU = 0.3
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.SCHEDULER.WARM_EPOCH = 30
cfg.TRAIN.SCHEDULER.MILESTONES = [200, 250, 290]
cfg.TRAIN.SCHEDULER.GAMMA = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.CONTEXT_GAP = None
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["GOT10K_vottrain"]#["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.VAL
cfg.DATA.VALTRACK = edict()
cfg.DATA.VALTRACK.DATASETS_NAME = ["OTB99_test"]
cfg.DATA.VALTRACK.DATASETS_RATIO = [1]
cfg.DATA.VALTRACK.SAMPLE_PER_EPOCH = 10000
# DATA.VAL
cfg.DATA.VALVL = edict()
cfg.DATA.VALVL.DATASETS_NAME = ["OTB99_test"]
cfg.DATA.VALVL.DATASETS_RATIO = [1]
cfg.DATA.VALVL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.CENTER_JITTER_GROUNDING = 4.5
cfg.DATA.SEARCH.SCALE_JITTER_GROUNDING = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.MODE = "NL"
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.THRESHOLD = 0.5
cfg.TEST.THRESHOLD_CONT = 0.0
cfg.TEST.THRESHOLD_CLS = 0.0
cfg.TEST.WINDOW_INFLUENCE = 0.49
cfg.TEST.UPDATE_INTERVAL = 100000
cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.LASOT = [200]
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = [200]
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = [200]
cfg.TEST.UPDATE_INTERVALS.VOT20 = [200]
cfg.TEST.UPDATE_INTERVALS.VOT20LT = [200]


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
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


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


