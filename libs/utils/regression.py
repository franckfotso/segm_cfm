# Project: segm_cfm
# Module: libs.utils.regression
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Based on:
#    py-faster-rcnn: https://github.com/rbgirshick/py-faster-rcnn
#    MNC: https://github.com/daijifeng001/MNC
# Licensed under MIT License

import os
import os.path as osp
import scipy.io
import numpy as np
from transformation import *
from cython_bbox import bbox_overlaps

def get_bbox_regression_labels(bbox_target_data, num_classes, cfg):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        '''
        print ("cls: {}").format(cls)
        print ("ind: {}").format(ind)
        print ("start: {}").format(start)
        print ("end: {}").format(end)
        print bbox_target_data[ind, 1:]
        print ("bbox_targets[ind]: {}").format(bbox_targets[ind])
        print bbox_targets[ind, start:end]
        #'''
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN_DEFAULT_BBOX_INSIDE_WEIGHTS
        
    return bbox_targets, bbox_inside_weights
    
def get_mask_regression_labels(mask_target_data, num_classes, cfg):
    pass

def compute_bbox_targets(ex_rois, gt_rois, labels, normalize, cfg):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_TARGETS_PRECOMPUTED and normalize:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_STDS))
    targets = np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
    
    return targets

def compute_bbox_means_stds(cache_im_dir, num_cls, cfg):
    """
    Compute bbox mean and stds for mcg proposals
    Since mcg proposal are stored on disk, so we precomputed it here once
    and save them to disk to avoid disk I/O next time
    Args:
        roidb_dir: directory contain all the mcg proposals
    """
    img_mat_fns = sorted(os.listdir(cache_im_dir))
    all_bbox_targets = []
    cnt = 0
    for img_mat_fn in img_mat_fns:
        img_mat_pn = os.path.join(cache_im_dir, img_mat_fn)
        try:
            img_mat = scipy.io.loadmat(img_mat_pn)
            bbox_targets = img_mat['sample_rois']['bbox_target_data'][0,0]
            all_bbox_targets.append(bbox_targets)
            cnt += 1
            del img_mat
        except:
            print ('[WARM] unable to read file: {}'.format(img_mat_fn))
            #raise Exception('[ERROR] unable to read file: {}'.format(img_mat_pn))
            

    class_counts = np.zeros((num_cls, 1)) + cfg.MAIN_DEFAULT_EPS
    sums = np.zeros((num_cls, 4))
    squared_sums = np.zeros((num_cls, 4))
    for im_i in xrange(len(all_bbox_targets)):
        bbox_targets = all_bbox_targets[im_i]
        for cls in xrange(1, num_cls):
            cls_inds = np.where(bbox_targets[:, 0] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += bbox_targets[cls_inds, 1:].sum(axis=0)
                squared_sums[cls, :] += \
                    (bbox_targets[cls_inds, 1:] ** 2).sum(axis=0)
    
    cache_dir = osp.join(cfg.MAIN_DIR_ROOT,cfg.MAIN_DIR_CACHE)
    means_file = osp.join(cache_dir,'{}_bbox_means.npy'\
                           .format(cfg.TRAIN_DEFAULT_SEGM_METHOD))
    stds_file = osp.join(cache_dir, '{}_bbox_stds.npy'\
                           .format(cfg.TRAIN_DEFAULT_SEGM_METHOD))
    
    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)
    np.save(means_file, means)
    np.save(stds_file, stds)
    
    return means, stds



    
    
    