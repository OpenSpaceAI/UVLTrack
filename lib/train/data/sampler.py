import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
from .utils import SimpleTokenizer
from pkg_resources import packaging
from pytorch_pretrained_bert import BertTokenizer

def no_processing(data):
    return data


class GroundingAndTrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    [base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, grounding_processing=None, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5, bert_path=None, mode='joint', grounding_ratio=None, vl_ratio=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.train_lang = False
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification
        self.mode = mode
        if mode == 'joint':
            assert grounding_ratio is not None
            assert vl_ratio is not None
            self.p_tracking = 1-grounding_ratio-vl_ratio
            self.p_vl = vl_ratio
            self.p_grounding = grounding_ratio
        elif mode == 'tracking':
            self.p_tracking = 1.0
            self.p_vl = 0.0
            self.p_grounding = 0.0
        elif mode == 'grounding':
            self.p_tracking = 0.0
            self.p_vl = 0.0
            self.p_grounding = 1.0

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.num_grounding_frames = 1
        self.processing = processing
        self.grounding_processing = grounding_processing
        self.frame_sample_mode = frame_sample_mode
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        
        # Datasets for different tasks
        self.tracking_dataset = [d for d in self.datasets if d.is_tracking_sequence()]
        self.p_tracking_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_tracking_sequence()]
        
        self.grounding_dataset = [d for d in self.datasets if d.is_grounding_sequence()]
        self.p_grounding_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_grounding_sequence()]
        
        self.vl_dataset = [d for d in self.datasets if d.is_vl_sequence()]
        self.p_vl_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_vl_sequence()]
        
    def __len__(self):
        if self.mode == "grounding_test":
            return self.datasets[0].get_num_sequences()
        else:
            return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        # Validation
        if self.mode == "grounding_test":
            return self.sample_grounding_test(index)
        elif self.mode == "tracking_test":
            return self.sample_track_test()
        elif self.mode == "vl_test":
            return self.sample_vl_test()
        # Train
        elif self.mode == 'tracking':
            return self.sample_track()
        elif self.mode == 'grounding':
            return self.sample_grounding()
        elif self.mode == 'joint':
            # Sample different modal samples for training
            seed = random.random()
            if seed < self.p_tracking:
                return self.sample_track()
            elif seed < self.p_tracking + self.p_grounding:
                return self.sample_grounding()
            else:
                return self.sample_vl()
        else:
            raise ValueError(f"No {self.mode} mode!")

    def sample_track(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.tracking_dataset, self.p_tracking_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()

            # Sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
                
            template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language*2,
                                'text_mask': mask*2,
                                'flag': torch.tensor([[0]])})
            data = self.processing.track_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_vl(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.vl_dataset, self.p_vl_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()

            # Sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
                
            template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language*2,
                                'text_mask': mask*2,
                                'flag': torch.tensor([[2]])})
            data = self.processing.track_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_grounding(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.grounding_dataset, self.p_grounding_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()
            # Sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            if is_video_dataset:
                grounding_frame_ids = None
                search_frame_ids = None
                gap_increase = 0
                MAX_N = 30
                while search_frame_ids is None:
                    if len(visible) < MAX_N:
                        MAX_N = len(visible)
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1,
                                                             min_id=self.num_grounding_frames - 1, 
                                                             max_id=MAX_N - self.num_search_frames + 1)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_grounding_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    grounding_frame_ids = base_frame_id + prev_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, min_id=grounding_frame_ids[0] + 1, max_id=
                                                                grounding_frame_ids[0] + self.max_gap + gap_increase, 
                                                                num_ids=self.num_search_frames-1)
                    # Increase gap until a frame is found
                    gap_increase += 5
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                grounding_frame_ids = [1] * self.num_grounding_frames
                search_frame_ids = [1] * (self.num_search_frames - 1)
                
            grounding_frames, grounding_anno, meta_obj_train = dataset.get_frames(seq_id, grounding_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'grounding_images': grounding_frames,
                                'grounding_anno': grounding_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language, # left right top bottom middle 2187, 2157, 2327, 3953, 2690
                                'text_mask': mask*2,
                                'flag': torch.tensor([[1]])})
            data = self.processing.grounding_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_vl_test(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            track_dataset = [d for d in self.datasets if d.is_video_sequence()]
            p_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_video_sequence()]
            dataset = random.choices(track_dataset, p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # Sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
                
            template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language*2,
                                'text_mask': mask*2,
                                'flag': torch.tensor([[2]])}) # torch.randint(2, (1,1))*2})
            data = self.processing.track_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_track_test(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            track_dataset = [d for d in self.datasets if d.is_video_sequence()]
            p_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_video_sequence()]
            dataset = random.choices(track_dataset, p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # Sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
                
            template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language*2,
                                'text_mask': mask*2,
                                'flag': torch.tensor([[0]])}) # torch.randint(2, (1,1))*2})
            data = self.processing.track_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_grounding_test(self, i):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        # Select a dataset
        valid = False
        while not valid:
            dataset = self.datasets[0]
            seq_id, visible, seq_info_dict = self.get_seq_from_dataset_by_id(dataset, i)
            grounding_frame_ids = [0]
            grounding_frames, grounding_anno, meta_obj_train = dataset.get_frames(seq_id, grounding_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'grounding_images': grounding_frames,
                                'grounding_anno': grounding_anno['bbox'],
                                'text': language,
                                'text_mask': mask,
                                'flag': torch.tensor([[1]])})
            data = self.processing.grounding_process(data)
            valid = data['valid']
        return data

    def get_center_box(self, H, W, ratio=1 / 8):
        cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
        return torch.tensor([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_seq_from_dataset_by_id(self, dataset, seq_id):
        seq_id = random.randint(0, dataset.get_num_sequences() - 1)
        seq_info_dict = dataset.get_sequence_info(seq_id)
        visible = seq_info_dict['visible']
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # Sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # Sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # Get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # Get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # Get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # First randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # Get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids
    
    def extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # Build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return [torch.tensor(input_ids)], [torch.tensor(input_mask)]


_tokenizer = SimpleTokenizer()

def tokenize(texts, context_length: int = 64, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    mask   = torch.ones(len(all_tokens), context_length+1, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            return None, None
        result[i, :len(tokens)] = torch.tensor(tokens)
        mask[i, :len(tokens)+1] = 0

    return result, mask.bool()
