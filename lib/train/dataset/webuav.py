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


class WebUAV(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None):
        root = env_settings().webuav_dir if root is None else root
        super().__init__('WebUAV', root, image_loader)

        self.sequence_list = self._build_sequence_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        seq_path = glob.glob(os.path.join(self.root, 'train/Train/', '*/'))
        sequence_list = [p.split('/')[-2] for p in seq_path]
        return sequence_list

    def get_name(self):
        return 'tnl2k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def is_grounding_sequence(self):
        return True

    def is_tracking_sequence(self):
        return True

    def is_vl_sequence(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absent.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, "train/Train", seq_name), seq_name

    def _read_language(self, seq):
        language_file = os.path.join(self.root, 'language/Language/Train', seq, "language.txt")
        with open(language_file, 'r') as f:
            language = f.readlines()
        return language[0].rstrip()

    def get_sequence_info(self, seq_id):
        seq_path, seq_name = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        images = sorted(glob.glob(os.path.join(seq_path, 'img', '*')))
        return self.image_loader(images[frame_id])

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path, seq_name = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        anno = self.get_sequence_info(seq_id)

        language = self._read_language(seq_name)
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
