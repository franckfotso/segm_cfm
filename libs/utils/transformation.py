# Project: segm_cfm
# Module: libs.utils.transformation
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Based on: py-faster-rcnn 
#    (https://github.com/rbgirshick/py-faster-rcnn)
# Licensed under MIT License

import cv2, math
import numpy as np
from utils.cython_bbox import bbox_overlaps
from nms.nms_wrapper import nms
from nms.mv import mv

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
        
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    
    return boxes

def filter_small_boxes(boxes, min_size):
    """
    Remove all boxes with any side smaller than min_size.
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def mask_overlap(box1, box2, mask1, mask2):
    """
    This function calculate region IOU when masks are
    inside different boxes
    Returns:
        intersection over unions of this two masks
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 > x2 or y1 > y2:
        return 0
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # get masks in the intersection part
    start_ya = y1 - box1[1]
    start_xa = x1 - box1[0]
    inter_maska = mask1[start_ya: start_ya + h, start_xa:start_xa + w]

    start_yb = y1 - box2[1]
    start_xb = x1 - box2[0]
    inter_maskb = mask2[start_yb: start_yb + h, start_xb:start_xb + w]

    assert inter_maska.shape == inter_maskb.shape, \
    'inter_maska.shape: {}, inter_maskb.shape: {}'.format(inter_maska.shape, inter_maskb.shape)

    inter = np.logical_and(inter_maskb, inter_maska).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)


def intersect_mask(ex_box, gt_box, gt_mask, mask_size, cfg):
    """
    This function calculate the intersection part of a external box
    and gt_box, mask it according to gt_mask

    Args:
        ex_box: external ROIS
        gt_box: ground truth boxes
        gt_mask: ground truth masks, not been resized yet
    Returns:
        regression_target: logical numpy array
    """
    x1 = max(ex_box[0], gt_box[0])
    y1 = max(ex_box[1], gt_box[1])
    x2 = min(ex_box[2], gt_box[2])
    y2 = min(ex_box[3], gt_box[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    
    mask_shape = (mask_size, mask_size)
        
    if x1 >= x2 or y1 >= y2:
        return np.zeros(mask_shape, dtype=np.int)
        
    inter_maskb = gt_mask[y1:y1+h, x1:x1+w]
    regression_target = np.zeros(gt_mask.shape)    
    regression_target[y1:y1+h, x1:x1+w] = inter_maskb
    regression_target = regression_target[ex_box[1]:ex_box[3]+1,ex_box[0]:ex_box[2]+1]
    regression_target = cv2.resize(regression_target, mask_shape)
    regression_target = (regression_target >= cfg.MAIN_DEFAULT_BINARIZE_THRESH).astype(bool)
    
    return regression_target


def pool_mask1(mask, box, mask_shape):
    """
        This function perform a spatial pyramid pooling over the mask
        target so that we get a fixed size for all masks: eg. 21x21
        Args:
            mask: mask with image size
            box: proposal bounding box (x1,y1,x2,y2)
            mask_shape: mask shape use for prediction
            cfg: config object with all settings
        Returns:
            mask_pooled: the mask pooled according mask_shape
    
    """    
    w = (box[2]-box[0])
    h = (box[3]-box[1])
    
    row_length = w / float(mask_shape[1])
    col_length = h / float(mask_shape[0])
    
    mask_pooled = []
    
    for jy in range(mask_shape[0]):
        for ix in range(mask_shape[1]):
            x1 = int(box[0] + ix * row_length)
            x2 = int(x1 + row_length)
            y1 = int(box[1] + jy * col_length)
            y2 = int(y1 + col_length)
            
            """
            x2 = x1 + max(1,x2-x1)
            y2 = y1 + max(1,y2-y1)
            """
            
            #"""
            if y2 < mask.shape[0]:
                y2 = y1 + max(1,y2-y1)
            else:
                y2 = mask.shape[0]
                y1 = y2 - 1
                
            if x2 < mask.shape[1]:
                x2 = x1 + max(1,x2-x1)
            else:
                x2 = mask.shape[1]
                x1 = x2 - 1
            #"""
            
            mask_crop = mask[y1:y2, x1:x2]
            mask_crop = np.max(mask_crop, axis=(0, 1))
            mask_pooled.append(mask_crop)
            
    mask_pooled = np.reshape(np.array(mask_pooled), mask_shape)
    
    return mask_pooled


def unpool_mask(mask, tgt_shape):
    #print ('mask.shape: {}'.format(mask.shape))
    #print ('bbox: {}'.format(bbox))
    
    tgt_h, tgt_w = tgt_shape
    cur_h, cur_w = mask.shape
    
    col_length = int(math.ceil(float(tgt_w) / float(cur_w)))
    row_length = int(math.ceil(float(tgt_h) / float(cur_h)))
    
    mask_unpooled = np.zeros((tgt_h, tgt_w))
    
    for jy in range(cur_h):
        for ix in range(cur_w):
            x1 = ix * col_length
            y1 = jy * row_length
            x2 = x1 + col_length      
            y2 = y1 + row_length
            
            if x1 >= mask_unpooled.shape[1] or \
               y1 >= mask_unpooled.shape[0]:
                    continue
            
            if y2 < mask_unpooled.shape[0]:
                y2 = y1 + max(1,y2-y1)
            else:
                y2 = y1 + (mask_unpooled.shape[0]-y1)
                
            if x2 < mask_unpooled.shape[1]:
                x2 = x1 + max(1,x2-x1)
            else:
                x2 = x1 + (mask_unpooled.shape[1]-x1)
            
            mask_unpooled[y1:y2,x1:x2] = mask[jy,ix]
            
    return mask_unpooled

def gpu_mask_voting(masks, boxes, scores, num_classes, max_per_image, im_width, im_height, cfg):
    """
    A wrapper function, note we already know the class of boxes and masks
    Args:
        masks: ~ 300 x 21 x 21
        boxes: ~ 300 x 4
        scores: ~ 300 x 1
        max_per_image: default would be 100
        im_width:
        im_height:
    """
    # Intermediate results
    sup_boxes = []
    sup_scores = []
    tobesort_scores = []
    for i in xrange(num_classes):
        if i == 0:
            sup_boxes.append([])
            sup_scores.append([])
            continue
        dets = np.hstack((boxes.astype(np.float32), scores[:, i:i+1].astype(np.float32)))
        #thresh = (cfg.TEST_DEFAULT_MASK_MERGE_IOU_THRESH).astype(np.float32)
        #print ('dets.shape: {}'.format(dets.shape))
        #print ('dets.dtype: {}'.format(dets.dtype))
        inds = nms(dets, cfg.TEST_DEFAULT_MASK_MERGE_IOU_THRESH, cfg)
        ind_boxes = boxes[inds]
        ind_scores = scores[inds, i]
        num_keep = min(len(ind_scores), max_per_image)
        sup_boxes.append(ind_boxes[0:num_keep, :])
        sup_scores.append(ind_scores[0:num_keep])
        tobesort_scores.extend(ind_scores[0:num_keep])

    sorted_scores = np.sort(tobesort_scores)[::-1]
    num_keep = min(len(sorted_scores), max_per_image)
    thresh = sorted_scores[num_keep-1]
    # inds array to record which mask should be aggregated together
    candidate_inds = []
    # weight for each element in the candidate inds
    candidate_weights = []
    # start position for candidate array
    candidate_start = []
    candidate_scores = []
    class_bar = []
    for c in xrange(num_classes):
        if c == 0:
            continue
        cls_box = sup_boxes[c]
        cls_score = sup_scores[c]
        keep = np.where(cls_score >= thresh)[0]
        new_sup_boxes = cls_box[keep]
        num_sup_box = len(new_sup_boxes)
        
        for i in xrange(num_sup_box):
            cur_ov = bbox_overlaps(boxes.astype(np.float), new_sup_boxes[i, np.newaxis].astype(np.float))
            cur_inds = np.where(cur_ov >= cfg.TEST_DEFAULT_MASK_MERGE_IOU_THRESH)[0]
            candidate_inds.extend(cur_inds)
            cur_weights = scores[cur_inds, c]
            cur_weights = cur_weights / sum(cur_weights)
            candidate_weights.extend(cur_weights)
            candidate_start.append(len(candidate_inds))
        candidate_scores.extend(cls_score[keep])
        class_bar.append(len(candidate_scores))
        
    candidate_inds = np.array(candidate_inds, dtype=np.int32)
    candidate_weights = np.array(candidate_weights, dtype=np.float32)
    candidate_start = np.array(candidate_start, dtype=np.int32)
    candidate_scores = np.array(candidate_scores, dtype=np.float32)
    
    #print ('boxes.shape: {}'.format(boxes.shape))
    #print ('masks.shape: {}'.format(masks.shape))
    masks = np.reshape(masks, (masks.shape[0],1,masks.shape[1],masks.shape[2])) # rfm add
    result_mask, result_box = mv(boxes.astype(np.float32), masks.astype(np.float32), 
                                 candidate_inds, candidate_start, 
                                 candidate_weights, im_height, im_width)
    #print ('result_mask.shape: {}'.format(result_mask.shape))
    #print ('result_box.shape: {}'.format(result_box.shape))
    result_box = np.hstack((result_box, candidate_scores[:, np.newaxis]))
    list_result_box = []
    list_result_mask = []
    # separate result mask into different classes
    for i in xrange(num_classes - 1):
        cls_start = class_bar[i - 1] if i > 0 else 0
        cls_end = class_bar[i]
        list_result_box.append(result_box[cls_start:cls_end, :])
        list_result_mask.append(result_mask[cls_start:cls_end, :, :, :])

    return list_result_mask, list_result_box


def mask_aggregation(boxes, masks, mask_weights, im_width, im_height, cfg):
    """
    This function implements mask voting mechanism to give finer mask
    n is the candidate boxes (masks) number
    Args:
        masks: All masks need to be aggregated (n x sz x sz)
        mask_weights: class score associated with each mask (n x 1)
        boxes: tight box enclose each mask (n x 4)
        im_width, im_height: image information
    TODO: Ensure mask size is sz x sz or tight box size
    """
    assert boxes.shape[0] == len(masks) and boxes.shape[0] == mask_weights.shape[0]
    im_mask = np.zeros((im_height, im_width))
    for mask_ind in xrange(len(masks)):
        box = np.round(boxes[mask_ind]).astype(np.int)
        mask = (masks[mask_ind] >= cfg.MAIN_DEFAULT_BINARIZE_THRESH).astype(float)
        mask_weight = mask_weights[mask_ind]
        im_mask[box[1]:box[3]+1, box[0]:box[2]+1] += mask * mask_weight
    [r, c] = np.where(im_mask >= cfg.MAIN_DEFAULT_BINARIZE_THRESH)
    if len(r) == 0 or len(c) == 0:
        min_y = int(np.ceil(im_height / 2))
        min_x = int(np.ceil(im_width / 2))
        max_y = min_y
        max_x = min_x
    else:
        min_y = np.min(r)
        min_x = np.min(c)
        max_y = np.max(r)
        max_x = np.max(c)

    clipped_mask = im_mask[min_y:max_y+1, min_x:max_x+1]
    clipped_box = np.array((min_x, min_y, max_x, max_y), dtype=np.float32)
    
    return clipped_mask, clipped_box


def cpu_mask_voting(masks, boxes, scores, num_classes, max_per_image, im_width, im_height, cfg):
    """
    Wrapper function for mask voting, note we already know the class of boxes and masks
    Args:
        masks: ~ n x mask_sz x mask_sz
        boxes: ~ n x 4
        scores: ~ n x 1
        max_per_image: default would be 100
        im_width: width of image
        im_height: height of image
    """
    # apply nms and sort to get first images according to their scores
    scores = scores[:, 1:] # remove bg scores
    num_detect = boxes.shape[0]
    res_mask = [[] for _ in xrange(num_detect)]
    for i in xrange(num_detect):
        box = np.round(boxes[i]).astype(int)
        mask = cv2.resize(masks[i].astype(np.float32), (box[2]-box[0]+1, box[3]-box[1]+1))
        # unpool mask pooled
        #mask = unpool_mask(masks[i], (box[3]-box[1]+1, box[2]-box[0]+1))
        res_mask[i] = mask
    # Intermediate results
    sup_boxes = []
    sup_masks = []
    sup_scores = []
    tobesort_scores = []

    for i in xrange(num_classes - 1):
        dets = np.hstack((boxes.astype(np.float32), scores[:, i:i+1]))
        inds = nms(dets, cfg.TEST_DEFAULT_MASK_MERGE_NMS_THRESH, cfg)
        ind_boxes = boxes[inds]
        ind_masks = masks[inds]
        ind_scores = scores[inds, i]
        order = ind_scores.ravel().argsort()[::-1]
        num_keep = min(len(order), max_per_image)
        order = order[0:num_keep]
        sup_boxes.append(ind_boxes[order])
        sup_masks.append(ind_masks[order])
        sup_scores.append(ind_scores[order])
        tobesort_scores.extend(ind_scores[order])

    sorted_scores = np.sort(tobesort_scores)[::-1]
    num_keep = min(len(sorted_scores), max_per_image)
    thresh = sorted_scores[num_keep-1]
    result_box = []
    result_mask = []
    for c in xrange(num_classes - 1):
        cls_box = sup_boxes[c]
        cls_score = sup_scores[c]
        keep = np.where(cls_score >= thresh)[0]
        new_sup_boxes = cls_box[keep]
        num_sup_box = len(new_sup_boxes)
        #masks_ar = np.zeros((num_sup_box, 1, cfg.MAIN_DEFAULT_MASK_SIZE, cfg.MAIN_DEFAULT_MASK_SIZE))
        masks_ar = np.zeros((num_sup_box, cfg.MAIN_DEFAULT_MASK_SIZE, cfg.MAIN_DEFAULT_MASK_SIZE))
        boxes_ar = np.zeros((num_sup_box, 4))
        for i in xrange(num_sup_box):
            # Get weights according to their segmentation scores
            cur_ov = bbox_overlaps(boxes.astype(np.float), new_sup_boxes[i, np.newaxis].astype(np.float))
            cur_inds = np.where(cur_ov >= cfg.TEST_DEFAULT_MASK_MERGE_IOU_THRESH)[0]
            cur_weights = scores[cur_inds, c]
            cur_weights = cur_weights / sum(cur_weights)
            # Re-format mask when passing it to mask_aggregation
            pass_mask = [res_mask[j] for j in list(cur_inds)]
            # do mask aggregation
            tmp_mask, boxes_ar[i] = mask_aggregation(boxes[cur_inds], pass_mask, cur_weights, im_width, im_height, cfg)
            tmp_mask = cv2.resize(tmp_mask.astype(np.float32), (cfg.MAIN_DEFAULT_MASK_SIZE, cfg.MAIN_DEFAULT_MASK_SIZE))
            # pool mask to get a fixed size
            #tmp_mask = pool_mask(tmp_mask, boxes_ar[i], (cfg.MAIN_DEFAULT_MASK_SIZE, cfg.MAIN_DEFAULT_MASK_SIZE))
            masks_ar[i] = tmp_mask
        # make new array such that scores is the last dimension of boxes
        boxes_scored_ar = np.hstack((boxes_ar, cls_score[keep, np.newaxis]))
        result_box.append(boxes_scored_ar)
        result_mask.append(masks_ar)
        
    return result_mask, result_box 

