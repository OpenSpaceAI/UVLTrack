import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='uvltrack', choices=['uvltrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline_base', help='yaml configure file name')
    args = parser.parse_args()

    return args

def evaluate_speed(model, template, search, text, prompt, flag):
    '''Speed Test'''
    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_test(template, search, text, prompt, flag)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_test(template, search, text, prompt, flag)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    dim = cfg.MODEL.HIDDEN_DIM

    if args.script == "uvltrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.uvltrack.build_model
        model = model_constructor(cfg)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        text = NestedTensor(torch.ones(bs, 40).long(), torch.randn(bs, 40)>0.5)
        prompt = torch.randn(bs, 3, dim)
        flag = torch.ones(bs).long()
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        text = text.to(device)
        prompt = prompt.to(device)
        flag = flag.to(device)
        evaluate_speed(model, template, search, text, prompt, flag)

    else:
        raise NotImplementedError
