import _init_paths
import argparse
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings
import glob

parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_name', type=str, help='Name of tracking method.')
parser.add_argument('--tracker_param', type=str, help='Name of config file.')
parser.add_argument('--dataset_name', type=str, help='Name of config file.')
parser.add_argument('--save_file', type=str, default=None)

args = parser.parse_args()

def check_complete(path):
    file_num = {
        'nfs': 200,
        'uav': 246,
        'lasotext': 300,
        'lasot': 560,
        'trackingnet': 1022,
        'tnl2k': 1400,
        'otb99': 96,
        'itb': 360,
        'avist': 240,
    }
    num_file = len(glob.glob(os.path.join(path, args.dataset_name, '*.txt')))
    for name, num in file_num.items():
        if name in args.dataset_name:
            if num_file == file_num[name]:
                return True
            else:
                return False
    raise ValueError("no such dataset")
            
env = env_settings()
trackers = []
tracker_params = [path.split('/')[-1] for path in sorted(glob.glob(os.path.join(env.results_path, args.tracker_name, args.tracker_param)), reverse=True) if check_complete(path)]
trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_param, dataset_name=args.dataset_name,
                            run_ids=None, display_name=args.tracker_name))

dataset = get_dataset(args.dataset_name)
print_results(trackers, dataset, report_name=args.dataset_name, merge_results=True, force_evaluation=True, plot_types=('success', 'prec', 'norm_prec'), save_file=args.save_file)
