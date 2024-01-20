import os
import os.path
import glob
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class OTB99(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None):
        root = env_settings().lasot_dir if root is None else root
        super().__init__('OTB99', root, image_loader)
        self.split = split
        self.sequence_list = self._build_sequence_list(split=split)

    def _build_sequence_list(self, vid_ids=None, split=None):
        seq_path = glob.glob(os.path.join(self.root, f'OTB_query_{split}/*.txt'))
        sequence_list = [p.split('/')[-1].split('.')[0] for p in seq_path]
        return sequence_list

    def get_name(self):
        return 'otb99'

    def is_grounding_sequence(self):
        return True

    def is_vl_sequence(self):
        return True

    def is_tracking_sequence(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        try:
            gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        except:
            gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id].split('-')[0] if self.split=='train' else self.sequence_list[seq_id]
        return os.path.join(self.root, 'OTB_videos', seq_name)

    def _read_language(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        language_file = os.path.join(self.root, f'OTB_query_{self.split}', f"{seq_name}.txt")
        with open(language_file, 'r') as f:
            language = f.readlines()
        return language[0].rstrip()

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        images = sorted(glob.glob(os.path.join(seq_path, 'img', '*')))
        return self.image_loader(images[frame_id])

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        anno = self.get_sequence_info(seq_id)

        language = self._read_language(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'language': language.lower()})

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames
