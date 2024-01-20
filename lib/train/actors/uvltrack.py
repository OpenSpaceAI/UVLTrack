from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh, box_xywh_to_cxcywh_scale
import torch
from lib import registry
from lib.utils.box_ops import giou_loss, GaussWeightedLoss
from torch.nn.functional import l1_loss
from lib.utils.misc import NestedTensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt
import cv2

@registry.ACTORS.register("uvltrack")
class UVLTrackActor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, cfg):
        super().__init__(net)
        self.cfg = cfg
        self.build_loss(cfg)

    def build_loss(self, cfg):
        weight = torch.tensor([self.cfg.DATA.SEARCH.FACTOR**2, self.cfg.TRAIN.CTR_RATIO**2]).cuda()
        weight = weight / weight.sum()
        self.objective = {'giou': giou_loss, 'l1': l1_loss, 
                          'cls': GaussWeightedLoss(reduction=cfg.TRAIN.REDUCTION),
                          'aux': CrossEntropyLoss(), 
                          'cib': CrossEntropyLoss(ignore_index=-1), 
                          'cont': CrossEntropyLoss(ignore_index=-1, weight=weight)}
        self.loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 
                            'cls': 1, 'aux': cfg.TRAIN.AUX_WEIGHT, 
                            'cib': cfg.TRAIN.CIB_WEIGHT, 'cont': cfg.TRAIN.CONT_WEIGHT}

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # process the groundtruth
        n, b, hc, wc = data['search_cls'].shape
        gt_bboxes = data['search_anno'].reshape(n*b, -1)  # (Ns, batch, 4) (x1,y1,w,h)
        gt_cls = data['search_cls'].reshape(n*b, hc, wc)  # (Ns, batch, 4) (x1,y1,w,h)
        
        # compute losses
        size = data['search_images'].shape[-1]//16
        loss, status = self.compute_losses(out_dict, gt_bboxes, gt_cls, self.cont_gt(gt_bboxes, size))

        return loss, status

    def forward_pass(self, data):
        _, b, _, ht, wt = data['template_images'].shape
        n, b, _, hs, ws = data['search_images'].shape
        template_images = data['template_images'].repeat(n, 1, 1, 1, 1).reshape(n*b, 3, ht, wt)
        template_anno = data['template_anno'].repeat(n, 1, 1).reshape(n*b, -1)
        search_images = data['search_images'].reshape(n*b, 3, hs, ws)
        search_anno = data['search_anno'].reshape(n*b, -1)
        text = data['text'].reshape(n*b, -1)
        text_mask = data['text_mask'].reshape(n*b, -1)
        text = NestedTensor(text, text_mask)
        template_mask = self.anno2mask(template_anno, wt//16)
        context_mask = self.anno2mask(search_anno, ws//16, reverse=True)
        flag = data['flag'].repeat(n, 1, 1).reshape(n*b, -1)
        out_dict = self.net(template_images, search_images, text, template_mask, context_mask, flag)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict
    
    def cont_gt(self, gt_bboxes, size):
        bboxes = box_cxcywh_to_xyxy(box_xywh_to_cxcywh_scale(gt_bboxes, self.cfg.TRAIN.CTR_RATIO))*size
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask_c = (x_mask & y_mask)

        cx = torch.floor((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = torch.floor((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask_c[bid, cy, cx] = True
        
        bboxes = box_cxcywh_to_xyxy(box_xywh_to_cxcywh(gt_bboxes))*size
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask_t = 1-2*(x_mask & y_mask).long()
        mask_t[mask_c] = 0
        return mask_t.flatten(1)
        
    def anno2mask(self, gt_bboxes, size, reverse=False):
        bboxes = box_xywh_to_xyxy(gt_bboxes)*size # b, 4
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask)

        cx = torch.floor((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = torch.floor((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask[bid, cy, cx] = True
        
        if reverse:
            mask = torch.cat([mask[bid.shape[0]//2:], mask[:bid.shape[0]//2]], dim=0)
        return mask.flatten(1)
        
    def sample_negative(self, logits, gt_bboxes, size):
        bboxes = gt_bboxes # b, 4
        cood_1d = (torch.arange(size)+0.5) / size
        cood = cood_1d.unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda() # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask) # b, h, w
        mask = (mask.reshape(gt_bboxes.shape[0], -1))*(-1e9) # background == 1
        sample_logits = torch.sort(logits.reshape(gt_bboxes.shape[0], -1)+mask, descending=True, dim=-1).values[:, :9]
        return sample_logits
        
    def contractive_learning(self, logits, gt_bbox):  # b, n, sz, sz
        b, n, sz, sz = logits.shape
        logits = logits.reshape(-1, 1, sz, sz)
        gt_bbox = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, n, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        ctr = (gt_bbox[:, :2] + gt_bbox[:, 2:]).reshape(b*n, 1, 1, 2) / 2
        neg_logits = self.sample_negative(logits, gt_bbox, sz).to(logits)
        sample_points = ctr * 2 - 1
        pos_logits = F.grid_sample(logits, sample_points, padding_mode="border", align_corners=True).reshape(b*n, -1) # b, 1, 1, 10
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        target = torch.zeros(b*n).to(gt_bbox.device).long()
        return logits, target # check
        

    def compute_losses(self, pred_dict, gt_bbox, gt_cls, gt_cont):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes'] # b, 4, s
        pred_cls = pred_dict['cls_score']
        pred_cont = pred_dict['cont_score']
        B = pred_cont.shape[0]
        if self.loss_weight['aux'] > 0:
            pred_logits, target = self.contractive_learning(pred_dict['logits'], gt_bbox)
            aux_loss = self.objective['aux'](pred_logits, target)
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        cls_loss = self.objective['cls'](pred_cls, gt_cls)
        cont_loss = self.objective['cont'](pred_cont.reshape(-1, 2), gt_cont.reshape(-1))
        # weighted sum
        loss =  self.loss_weight['giou'] * giou_loss + \
                    self.loss_weight['l1'] * l1_loss + \
                        self.loss_weight['cls'] * cls_loss + \
                            self.loss_weight['aux'] * aux_loss + \
                                self.loss_weight['cont'] * cont_loss
        
        # status for log
        mean_iou = iou.detach().mean()
        status = {"Loss/total": loss,
                    "Loss/giou": giou_loss,
                    "Loss/l1": l1_loss,
                    "Loss/cls": cls_loss,
                    "Loss/aux": aux_loss,
                    "Loss/cont": cont_loss,
                    "IoU": mean_iou}
        if not self.net.training:
            acc = torch.mean((iou > 0.5).float()).detach()
            status['Acc@0.5'] = acc
        return loss, status
