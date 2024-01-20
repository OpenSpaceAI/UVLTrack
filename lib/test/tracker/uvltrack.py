from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, grounding_resize
# for debug
from copy import deepcopy
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.uvltrack.uvltrack import build_model
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, box_cxcywh_to_xywh, box_cxcywh_to_xyxy
import numpy as np
import matplotlib.pyplot as plt
from lib.test.utils.hann import hann2d

from pytorch_pretrained_bert import BertTokenizer
from lib.utils.misc import NestedTensor


class UVLTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(UVLTrack, self).__init__(params)
        network = build_model(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.map_size = params.search_size // 16
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = self.params.debug
        self.frame_id = 0
        # if self.debug:
        #     self.save_dir = "/ssd/myc/VL_project/MUTrack/debug"
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)
        self.update_interval = self.cfg.TEST.UPDATE_INTERVAL
        self.feat_size = self.params.search_size // 16
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH, do_lower_case=True)
        self.threshold = self.params.cfg.TEST.THRESHOLD
        self.has_cont = self.params.cfg.TRAIN.CONT_WEIGHT > 0
        self.max_score = 0

    def grounding(self, image, info: dict):
        bbox = torch.tensor([0., 0., 0., 0.]).cuda()
        h, w = image.shape[:2]
        im_crop_padded, _, _, _, _ = grounding_resize(image, self.params.grounding_size, bbox, None)
        ground = self.preprocessor.process(im_crop_padded).cuda()
        template = torch.zeros([1, 3, self.params.template_size, self.params.template_size]).cuda()
        template_mask = torch.zeros([1, (self.params.template_size//16)**2]).bool().cuda()
        context_mask = torch.zeros([1, (self.params.search_size//16)**2]).bool().cuda()
        text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)
        self.text = NestedTensor(text, mask)
        flag = torch.tensor([[1]]).cuda()
        with torch.no_grad():
            out_dict = self.network.forward(template, ground, self.text, template_mask, context_mask, flag)
        out_dict['pred_boxes'] = box_cxcywh_to_xywh(out_dict['pred_boxes']*np.max(image.shape[:2]))[0, 0].cpu().tolist()
        dx, dy = min(0, (w-h)/2), min(0, (h-w)/2)
        out_dict['pred_boxes'][0] = out_dict['pred_boxes'][0] + dx
        out_dict['pred_boxes'][1] = out_dict['pred_boxes'][1] + dy
        return out_dict

    def window_prior(self):
        hanning = np.hanning(self.map_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        self.torch_window = hann2d(torch.tensor([self.map_size, self.map_size]).long(), centered=True).flatten()

    def initialize(self, image, info: dict):
        if self.cfg.TEST.MODE == 'NL':
            grounding_state = self.grounding(image, info)
            init_bbox = grounding_state['pred_boxes']
            self.flag = torch.tensor([[2]]).cuda()
        elif self.cfg.TEST.MODE == 'NLBBOX':
            text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[2]]).cuda()
        else:
            text = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).long().cuda()
            mask = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).cuda()
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[0]]).cuda()
        self.window_prior()
        z_patch_arr, _, _, bbox = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size, return_bbox=True)
        self.template_mask = self.anno2mask(bbox.reshape(1, 4), size=self.params.template_size//16)
        self.z_patch_arr = z_patch_arr
        self.template_bbox = (bbox*self.params.template_size)[0, 0].tolist()
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        # forward the context once
        y_patch_arr, _, _, y_bbox = sample_target(image, init_bbox, self.params.search_factor,
                                                    output_sz=self.params.search_size, return_bbox=True)
        self.y_patch_arr = y_patch_arr
        self.context_bbox = (y_bbox*self.params.search_size)[0, 0].tolist()
        context = self.preprocessor.process(y_patch_arr)
        context_mask = self.anno2mask(y_bbox.reshape(1, 4), self.params.search_size//16)
        self.prompt = self.network.forward_prompt_init(self.template, context, self.text, self.template_mask, context_mask, self.flag)
        # save states
        self.state = init_bbox
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            template = self.template
            out_dict = self.network.forward_test(template, search, self.text, self.prompt, self.flag)

        pred_boxes = out_dict['bbox_map'].view(-1, 4).detach().cpu() # b, s, 4
        pred_cls = out_dict['cls_score_test'].view(-1).detach().cpu() # b, s
        pred_cont = out_dict['cont_score'].softmax(-1)[:, :, 0].view(-1).detach().cpu() if self.has_cont else 1 # b, s
        pred_cls_merge = pred_cls * self.window * pred_cont
        pred_box_net = pred_boxes[torch.argmax(pred_cls_merge)]
        score = (pred_cls * pred_cont)[torch.argmax(pred_cls_merge)]
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_box_net * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        if score > self.max_score and self.has_cont:
            self.pred_box_net = pred_box_net
            self.out_dict = out_dict
            self.max_score = score
        
        if self.frame_id % self.update_interval == 0 and self.has_cont and self.max_score > self.threshold:
            self.y_patch_arr = x_patch_arr
            context_bbox = box_cxcywh_to_xywh(self.pred_box_net.reshape(1, 4))
            context_mask = self.anno2mask(context_bbox, self.params.search_size//16)
            self.context_bbox = (context_bbox[0]*self.params.search_size).detach().cpu().tolist()
            self.prompt = self.network.forward_prompt(self.out_dict, self.template_mask, context_mask)
            self.max_score = 0
            
        return {"target_bbox": self.state}

    def save_visualization(self, image, vis_info):
        # save_name = os.path.join(self.save_dir, self.params.yaml_name+'_vis', vis_info['name'], '%.4d'%self.frame_id)
        save_name = self.save_dir
        if not os.path.exists(os.path.join(save_name)):
            os.makedirs(save_name)

        color = [(255, 0, 0), (0, 255, 0)]

        for img, name, bbox in zip(vis_info['patches'], vis_info['patches_name'], vis_info['patches_bbox']):
            x, y, w, h = bbox
            img = cv2.rectangle(img, (int(x), int(y)),(int(x+w), int(y+h)), color[0], 2)
            plt.imsave(os.path.join(save_name, f'{name}.png'), img)

        for i, img in enumerate(vis_info['cls_map']):
            img = cv2.resize(img.numpy(), (200, 200))
            plt.imsave(os.path.join(save_name, f'clsmap_{i}.png'), img)

        for i, vis_bbox in enumerate(vis_info['image_bbox']):
            x, y, w, h = vis_bbox
            image = cv2.rectangle(image, (int(x), int(y)),(int(x+w), int(y+h)), color[i], 2)
        scale = 400/max(image.shape[:2])
        dh, dw = image.shape[:2]
        image = cv2.resize(image, (int(dw*scale), int(dh*scale)))
        plt.imsave(os.path.join(save_name, 'image_bbox.jpg'), image)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
        
    def anno2mask(self, gt_bboxes, size):
        bboxes = box_xywh_to_xyxy(gt_bboxes)*size # b, 4
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1)+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask)

        cx = ((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = ((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask[bid, cy, cx] = True
        return mask.flatten(1).cuda()
    
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
        # build tokens and token_ids
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

        return torch.tensor(input_ids).unsqueeze(0).cuda(), torch.tensor(input_mask).unsqueeze(0).cuda()


def get_tracker_class():
    return UVLTrack
