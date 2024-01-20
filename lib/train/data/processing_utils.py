import torch
import math
import cv2
import copy
import cv2 as cv
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''
def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def grounding_resize(im, output_sz, bbox, mask=None):
    """ Resize the grounding image without change the aspect ratio, First choose the short side,then resize_factor =
    scale_factor * short side / long size, then padding the border with value 0

    args:
        im - cv image
        output_sz - return size of img int
        bbox - the bounding box of target in image , which form is (X, Y, W, H)
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        mask - the image of mask which size is [H, W] numpy array
    returns:
        im_crop_padded  - resized and padded image which shape is (resize_H, resize_W, C)
        box - resize and normalize, the coord is normalized to [0,1]
        att_mask - shape is (resize_H, resize_W)  the value of padding pixel is 1, the original pixel is 0
        mask_crop_padded - all zero and shape is (H, W)
    """
    h, w = im.shape[0:-1]
    scale_factor = 1
    crop_sz = math.ceil(scale_factor * output_sz)
    interpolation = Image.BILINEAR
    if w > h:
        ow = crop_sz
        oh = int(crop_sz * h / w)
    else:
        oh = crop_sz
        ow = int(crop_sz * w / h)
    # Resie image
    img = cv2.resize(im, (ow, oh), interpolation)

    new_h, new_w = img.shape[0:2]
    # Center padding
    y1_pad = int((output_sz - new_h) / 2)
    y2_pad = int((output_sz - new_h) / 2)
    x1_pad = int((output_sz - new_w) / 2)
    x2_pad = int((output_sz - new_w) / 2)
    # Bottom padding
    # y1_pad = 0
    # y2_pad = int((output_sz - new_h))
    # x1_pad = 0
    # x2_pad = int((output_sz - new_w))
    if (y1_pad + y2_pad + new_h) != output_sz:
        y1_pad += 1
    if (x1_pad + x2_pad + new_w) != output_sz:
        x1_pad += 1
    box = copy.deepcopy(bbox)

    # Scale the box size
    box[0] = bbox[0] * new_w / w
    box[1] = bbox[1] * new_h / h
    box[2] = bbox[2] * new_w / w
    box[3] = bbox[3] * new_h / h

    assert (y1_pad + y2_pad + new_h) == output_sz and (x1_pad + x2_pad + new_w) == output_sz, print(
        'y1_pad:{},y2_pad:{},x1_pad:{},x2_pad:{}'.format(y1_pad, y2_pad, x1_pad, x2_pad)) and print(
        f'img shape:{img.shape}')
    # The left top coord of the resized image in the padding image
    image_top_coords = [x1_pad, y1_pad, new_w, new_h]
    # Pad
    im_crop_padded = cv2.copyMakeBorder(img, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, (0, 0, 0))
    # Add the padding distance
    box[0] += x1_pad
    box[1] += y1_pad
    # Normalized to [0,1]
    box /= output_sz

    H, W, _ = im_crop_padded.shape
    if mask is not None:
        # todo find a better way to resize mask, mask is a tensor which all values is zero
        mask_crop_padded = torch.zeros(H, W)
        # mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)
    else:
        mask_crop_padded = torch.zeros(H, W)

    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    return im_crop_padded, box, att_mask, mask_crop_padded, image_top_coords

def generate_cls_label(bboxes, gaussian_iou=0.7, out_size=20, dynamic=False):
    cls_maps = []
    for bbox in bboxes:
        x, y, w, h = bbox*out_size
        cx, cy = int(x+w/2), int(y+h/2)
        if dynamic:
            radius = gaussian_radius((h, w), gaussian_iou)
            radius = max(0, int(radius))
        else:
            radius = 2.
        cls_map = np.zeros([out_size, out_size])
        draw_gaussian(cls_map, [cx, cy], radius)
        cls_map_save = cv2.resize(np.uint8(cls_map*255), (256, 256))
        cls_maps.append(torch.tensor(cls_map))
    return cls_maps

def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None, return_bbox=False):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    x2 = int(x1 + crop_sz)

    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    y2 = int(y1 + crop_sz)

    x1_pad = int(max(0, -x1))
    x2_pad = int(max(x2 - im.shape[1] + 1, 0))

    y1_pad = int(max(0, -y1))
    y2_pad = int(max(y2 - im.shape[0] + 1, 0))

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # Deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    bbox = torch.tensor([[[0.5-w/crop_sz/2, 0.5-h/crop_sz/2, w/crop_sz, h/crop_sz]]])
    if return_bbox:
        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
            att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
            if mask is None:
                return im_crop_padded, resize_factor, att_mask, bbox
            mask_crop_padded = \
            F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded, bbox

        else:
            if mask is None:
                return im_crop_padded, att_mask.astype(np.bool_), 1.0, bbox
            return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded, bbox
    else:
        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
            att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
            if mask is None:
                return im_crop_padded, resize_factor, att_mask
            mask_crop_padded = \
            F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded

        else:
            if mask is None:
                return im_crop_padded, att_mask.astype(np.bool_), 1.0
            return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

def gauss_1d(sz, sigma, center, end_pad=0):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    return torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)


def gauss_2d(sz, sigma, center, end_pad=(0, 0)):
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0]).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1]).reshape(center.shape[0], -1, 1)


def gaussian_label_function(target_bb, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=True):
    """Construct Gaussian label function."""

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)
    feat_sz = torch.Tensor(feat_sz)

    target_center = target_bb[:, 0:2] + 0.5 * target_bb[:, 2:4]
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad)

    return gauss_label


def perturb_box(box, min_iou=0.0, max_iou=0.3, sigma_factor=0.5):
    """ Perturb the input box by adding gaussian noise to the co-ordinates
     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate
    returns:
        torch.Tensor - the perturbed box
    """

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, torch.Tensor):
        c_sigma_factor = c_sigma_factor * torch.ones(4)

    perturb_factor = torch.sqrt(box[2] * box[3]) * c_sigma_factor

    # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5 * box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = random.gauss(c_x, perturb_factor[0])
        c_y_per = random.gauss(c_y, perturb_factor[1])

        w_per = random.gauss(box[2], perturb_factor[2])
        h_per = random.gauss(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2] * rand_uniform(0.15, 0.5)

        if h_per <= 1:
            h_per = box[3] * rand_uniform(0.15, 0.5)

        box_per = torch.Tensor([c_x_per - 0.5 * w_per, c_y_per - 0.5 * h_per, w_per, h_per]).round()

        if box_per[2] <= 1:
            box_per[2] = box[2] * rand_uniform(0.15, 0.5)

        if box_per[3] <= 1:
            box_per[3] = box[3] * rand_uniform(0.15, 0.5)

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # if there is sufficient overlap, return
        if box_iou > min_iou and box_iou < max_iou:
            return box_per, box_iou

        # else reduce the perturb factor
        perturb_factor *= 0.9

    return box_per, box_iou


def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.
    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)
    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = torch.max(reference[:, :2], proposals[:, :2])
    br = torch.min(reference[:, :2] + reference[:, 2:], proposals[:, :2] + proposals[:, 2:])
    sz = (br - tl).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = reference[:, 2:].prod(dim=1) + proposals[:, 2:].prod(dim=1) - intersection

    return intersection / union


def rand_uniform(a, b, shape=1):
    """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor
    returns:
        torch.Tensor - tensor of shape=shape
    """
    return (b - a) * torch.rand(shape) + a