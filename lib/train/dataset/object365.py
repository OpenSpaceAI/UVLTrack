import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import json
import torch
import random
from pycocotools.coco import COCO
from collections import OrderedDict
from lib.train.admin import env_settings
from .utils import generate_sentence

class Object365(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2014"):
        super().__init__('Object365', root, image_loader)

        self.img_pth = os.path.join(root, 'imgs/')
        self.anno_path = os.path.join(root, 'zhiyuan_objv2_train.json')
        self.sequence_list = self._get_sequence_list()
        self.id2class = {}
        for cat in self.region_descriptions['categories']:
            self.id2class[cat['id']] = cat['name']

    def _get_sequence_list(self):
        with open(self.anno_path, 'r') as f:
            self.region_descriptions = json.load(f)
        seq_list = list(range(len(self.region_descriptions['annotations'])))
        return seq_list

    def is_video_sequence(self):
        return False

    def is_grounding_sequence(self):
        return False

    def get_name(self):
        return 'object365'

    def has_class_info(self):
        return True

    def has_segmentation_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)
        bbox = torch.Tensor(anno['bbox']).view(1, 4)
        valid = torch.Tensor([True])
        visible = torch.Tensor([True])

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_anno(self, seq_id):
        desc = self.region_descriptions['annotations'][seq_id]
        anno = {
            'bbox': desc['bbox']
        }
        return anno

    def _get_frames(self, seq_id):
        desc = self.region_descriptions['annotations'][seq_id]
        img_path = os.path.join(self.img_pth, "objects365_v1_%08d.jpg"%(desc['image_id']))
        if os.path.exists(img_path):
            img = self.image_loader(img_path)
        else:
            img = self.image_loader(os.path.join(self.img_pth, "objects365_v2_%08d.jpg"%(desc['image_id'])))
        return img

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        language = self.id2class[self.region_descriptions['annotations'][seq_id]['category_id']]
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'language': generate_sentence(language.lower())})

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...].clone() for _ in frame_ids]

        return anno_frames