import os
import os.path as osp
import random
from collections import OrderedDict
import numpy as np

import torch

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from .refer import REFER

# Meta Information
SUPPORTED_DATASETS = {
    'otb': {'splits': ('test', 'train')},
    'tnl2k': {'splits': ('test', 'train')},
    'lasot': {'splits': ('test', 'train')},
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
    },
    'flickr': {
        'splits': ('train', 'val', 'test')}
}


class RefCOCOSeq(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2014",
                 name="refcocog", splitBy="google"):
        root = env_settings().coco_dir if root is None else root
        super().__init__('RefCOCOSeq', root, image_loader)
        self.split = split
        self.img_pth = os.path.join(root, '{}{}'.format("train", version))
        self.anno_path = os.path.join(root, '{}/instances.json'.format(name))
        self.dataset_name = name
        # Load the COCO set.
        self.coco_set = REFER(root, dataset=name, splitBy=splitBy)

        self.cats = self.coco_set.Cats

        self.class_list = self.get_class_list()

        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        self.seq_per_class = self._build_seq_per_class()
        self.im_dir, self.covert_bbox, self.img_names, self.phrases = self.get_split_info('gref')

    def _get_sequence_list(self):
        ref_list = list(self.coco_set.getRefIds(split=self.split))
        seq_list = [a for a in ref_list if self.coco_set.refToAnn[a]['iscrowd'] == 0]

        return seq_list

    def get_split_info(self, dataset):
        # setting datasource
        self.split_root = '/ssd/myc/VL_project/VLTVG/split/data'
        split = self.split
        im_dir = osp.join(self.root, 'train2014')

        dataset_split_root = osp.join(self.split_root, dataset)
        valid_splits = SUPPORTED_DATASETS[dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    dataset, split))

        # read the image set info
        self.imgset_info = []
        splits = [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(dataset, split)
            imgset_path = osp.join(dataset_split_root, imgset_file)
            self.imgset_info += torch.load(imgset_path, map_location="cpu")

        # process the image set info
        img_names, _, bboxs, phrases, _ = zip(*self.imgset_info)

        covert_bbox = []
        for bbox in bboxs:  # for referit, flickr
            bbox = np.array(bbox, dtype=np.float32)
            covert_bbox.append(bbox)
        return im_dir, covert_bbox, img_names, phrases

    def is_video_sequence(self):
        return False

    def is_grounding_sequence(self):
        return True

    def is_tracking_sequence(self):
        return False

    def is_vl_sequence(self):
        return True

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return self.dataset_name

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id])
        return class_list

    def has_segmentation_info(self):
        return True

    def get_num_sequences(self):
        return len(self.img_names)

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = self.cats[self.coco_set.refToAnn[seq]['category_id']]
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def get_sequence_info(self, seq_id):
        bbox = torch.Tensor(self.covert_bbox[seq_id]).view(1, 4)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _read_nlp(self, seq_id):

        ref = self.coco_set.Refs[self.sequence_list[seq_id]]
        sent = ref['sentences'][-1]['sent']

        return sent

    def _get_anno(self, seq_id):
        anno = self.coco_set.refToAnn[self.sequence_list[seq_id]]

        return anno

    def _get_frames(self, seq_id):
        img_path = os.path.join(self.im_dir, self.img_names[seq_id])
        img = self.image_loader(img_path)
        return img

    def get_meta_info(self, seq_id):
        nlp = self.phrases[seq_id]
        try:
            cat_dict_current = self.cats[self.coco_set.refToAnn[self.sequence_list[seq_id]]['category_id']]
            object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None,
                                       'language': nlp})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None,
                                       'language': nlp})
        return object_meta

    def get_class_name(self, seq_id):
        cat_dict_current = self.cats[self.coco_set.refToAnn[self.sequence_list[seq_id]]['category_id']]
        return cat_dict_current['name']

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta

    def get_path(self, seq_id, frame_ids):
        img_name = self.coco_set.loadImgs([self.coco_set.refToAnn[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        return [os.path.join(self.img_pth, img_name) for _ in range(len(frame_ids))]

    def get_ref_id(self, seq_id):
        return self.sequence_list[seq_id]