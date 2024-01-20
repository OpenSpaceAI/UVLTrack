import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import lib.train.data.processing_utils_grounding as prutils_grounding
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, 
                joint_transform=None, grounding_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'grounding': transform if grounding_transform is None else grounding_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError

class TrackProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, center_jitter_factor_grounding, scale_jitter_factor_grounding,
                 mode='pair', settings=None, train_score=False, dynamic_cls=False, gaussian_iou=0.7, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.center_jitter_factor_grounding = center_jitter_factor_grounding
        self.scale_jitter_factor_grounding = scale_jitter_factor_grounding
        self.mode = mode
        self.settings = settings
        self.train_score = train_score
        self.dynamic_cls = dynamic_cls
        self.gaussian_iou = gaussian_iou

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _get_jittered_box_grounding(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor_grounding)
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor_grounding).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_neg_proposals(self, box, min_iou=0.0, max_iou=0.3, sigma=0.5):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box
        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        num_proposals = box.size(0)
        proposals = torch.zeros((num_proposals, 4)).to(box.device)
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box[i], min_iou=min_iou, max_iou=max_iou,
                                                             sigma_factor=sigma)
        return proposals

    def track_process(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=None)
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], attn = self.transform[s](image=crops, bbox=boxes, att=att_mask, joint=False) # x, y, w, h, 0-1

            for ele in attn:
                if (ele == 1).all():
                    data['valid'] = False
                    return data
            for ele in attn:
                feat_size = self.output_sz[s] // 16
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    return data

            if s == 'search':
                data[s + '_cls'] = prutils.generate_cls_label(data[s + '_anno'], gaussian_iou=self.gaussian_iou, out_size=feat_size, dynamic=self.dynamic_cls)

        data['valid'] = True
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
        
    def has_directions(self, text):
        exclude_words = torch.tensor([[2187, 2157, 2327, 3953, 2690]])
        return ((text[0].unsqueeze(-1) - exclude_words) == 0).any()
        
    def grounding_process(self, data: TensorDict):
        """Generates proposals by adding noise to the input box
        args:
            data - The input data, should contain the following fields:
                'grounding_images', search_images', 'grounding_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'grounding_images', 'search_images', 'grounding_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        has_search = data.get('search_images', None) is not None
        has_direction = self.has_directions(data['text'])

        # For grounding image
        if has_search:
            grounding_resize = [prutils_grounding.grounding_resize(im, self.output_sz['grounding'], box, data['text'][0])
                                for im, box in zip(data['grounding_images'], data['grounding_anno'])]
            resize_grounding_frames, resize_grounding_box, grounding_att_mask, _, _, phrase_grounding = zip(*grounding_resize)
            
            search_resize = [prutils_grounding.grounding_resize(im, self.output_sz['search'], box, data['text'][0])
                                for im, box in zip(data['search_images'], data['search_anno'])]
            resize_search_frames, resize_search_box, search_att_mask, _, _, phrase_search = zip(*search_resize)
            data['text'] = phrase_grounding+phrase_search
            data['grounding_images'], data['grounding_anno'], data['grounding_att'] = \
                self.transform['grounding'](image=resize_grounding_frames, bbox=resize_grounding_box, att=grounding_att_mask, joint=False)
            data['search_images'], data['search_anno'], data['search_att'] = \
                self.transform['grounding'](image=resize_search_frames, bbox=resize_search_box, att=search_att_mask, joint=False)
        else:
            grounding_resize = [prutils.grounding_resize(im, self.output_sz['grounding'], box)
                                for im, box in zip(data['grounding_images'], data['grounding_anno'])]
            resize_grounding_frames, resize_grounding_box, grounding_att_mask, _, _ = zip(*grounding_resize)
            data['grounding_images'], data['grounding_anno'], data['grounding_att'] = \
                self.transform['grounding'](image=resize_grounding_frames, bbox=resize_grounding_box, att=grounding_att_mask, joint=False)
        
        iter_list = ['search', 'grounding'] if has_search else ['grounding']
        for s in iter_list:
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
            for ele in data[s + '_att']:
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    print("Values of down-sampled attention mask are all one. Replace it with new data.")
                    return data
            del data[s + '_att']
            data[s + '_cls'] = prutils.generate_cls_label(data[s + '_anno'], gaussian_iou=self.gaussian_iou, out_size=feat_size, dynamic=self.dynamic_cls)

        if not has_search:
            data['search_images'] = data['grounding_images']
            data['search_anno'] = data['grounding_anno']
            data['search_cls'] = data['grounding_cls']
        else:
            data['search_images'] = (data['grounding_images'] + data['search_images'])
            data['search_anno'] = (data['grounding_anno'] + data['search_anno'])
            data['search_cls'] = (data['grounding_cls'] + data['search_cls'])
            
        data['template_images'] = [torch.zeros([3, self.output_sz['template'], self.output_sz['template']])]
        data['template_anno'] = [torch.zeros([4])]
        del data['grounding_images']
        del data['grounding_anno']
        del data['grounding_cls']

        data['valid'] = True
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        return data