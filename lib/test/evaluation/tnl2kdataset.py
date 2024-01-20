import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os

class TNL2KDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_list = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'imgs', '*')))

        language_file = os.path.join(self.base_path, sequence_name, "language.txt")
        
        with open(language_file, 'r') as f:
            language = f.readlines()[0].rstrip()
        
        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
                        object_class=None, target_visible=None, language=language)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = sorted([p.split('/')[-2] for p in glob.glob(os.path.join(self.base_path, '*/'))])
        return sequence_list
