# Project: segm_cfm
# Module: main.TestWrapper
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Based on:
#    MNC: https://github.com/daijifeng001/MNC
# Licensed under MIT License

import caffe, os, heapq, cPickle, cv2

from metric.PascalEval import PascalEval
from nms.nms_wrapper import apply_nms_mask_single
from utils.timer import Timer
from utils.transformation import filter_small_boxes

import numpy as np
import os.path as osp
import scipy.io as sio


class TestWrapper(object):
    """
       A test wrapper involving tools to prepare data so that
       we can perform a net.forward and extract targeted outputs.
    """
    
    def __init__(self, test_prototxt, test_model, task, data_gen, output_dir, cfg):
        self.net = caffe.Net(test_prototxt, test_model, caffe.TEST)
        self.net.name = os.path.splitext(os.path.basename(test_model))[0]
        
        self.task = task
        self.data_gen = data_gen
        self.output_dir = output_dir
        self.cfg = cfg
        
        # heuristic: keep an average of 40 detections per class per images prior to nms
        self.max_per_set = 40 * len(data_gen.images)
        # heuristic: keep at most 100 detection per class per image prior to NMS
        self.max_per_image = 100
        
    
    def cfm_net_forward(self, cfm_t, image, backbone_net):
        im_nm = (image.filename).split('.')[0]
        pr_mat_pn = osp.join(self.data_gen.in_pr_dir, im_nm+'.mat')
        
        assert osp.exists(pr_mat_pn), \
        '[ERROR] missing proposals mat data for the image: {}'.format(pr_mat_pn)              
        pr_mat = sio.loadmat(pr_mat_pn)
        
        pr_labels = pr_mat['labels'][0]
        pr_bboxes = pr_mat['bboxes']
        pr_superpixels = pr_mat['superpixels']
        
        if len(pr_superpixels.shape) == 2:
            pr_masks = np.zeros((len(pr_labels),pr_superpixels.shape[0],
                                      pr_superpixels.shape[1]), dtype=np.float)
            for m_i, label in enumerate(pr_labels):
                [row_ids, col_ids] = np.array(np.where((pr_superpixels == label)))
                pr_masks[m_i,row_ids, col_ids] = 1
                
        elif len(pr_superpixels.shape) == 3:
            pr_masks = pr_superpixels
        else:
            raise Exception('[ERROR] incorrect shape for pr_superpixels: {}'\
                            .format(pr_superpixels.shape))     
        
        #print 'pr_bboxes.shape: {}'.format(pr_bboxes.shape)
        keep_inds = filter_small_boxes(pr_bboxes, min_size=self.cfg.TEST_DEFAULT_ROI_MIN_SIZE)
        pr_bboxes = pr_bboxes[keep_inds]        
        pr_masks = pr_masks[keep_inds]
        
        # pool mask bef. forward
        num_pr = pr_masks.shape[0]
        mask_in_shape = tuple(self.cfg.MAIN_DEFAULT_MASK_POOL_SHAPE)
        mask_out_shape = (self.cfg.MAIN_DEFAULT_MASK_SIZE, self.cfg.MAIN_DEFAULT_MASK_SIZE)
        pr_masks_pooled = np.zeros((num_pr,)+mask_in_shape)
        or_masks_pooled = np.zeros((num_pr,)+mask_out_shape)
        for m_i in xrange(num_pr):
            pr_box = pr_bboxes[m_i]
            pr_mask = pr_masks[m_i]
            mask_crop = pr_mask[pr_box[1]:pr_box[3]+1,pr_box[0]:pr_box[2]+1]
            pr_masks_pooled[m_i] = cv2.resize(mask_crop.astype(np.float), mask_in_shape, interpolation=cv2.INTER_NEAREST)
            or_masks_pooled[m_i] = cv2.resize(mask_crop.astype(np.float), mask_out_shape, interpolation=cv2.INTER_NEAREST)
        pr_masks = pr_masks_pooled
        #print 'pr_masks.shape: {}'.format(pr_masks.shape)
        #print 'mask_shape: {}'.format(mask_shape)
        
        assert pr_bboxes.shape[0] == pr_masks.shape[0], \
            '[ERROR] rois & masks size mismatch: {} vs. {}'.\
            format(pr_bboxes.shape[0], pr_masks.shape[0])
                        
        # Get top-k proposals
        if self.cfg.TEST_DEFAULT_TOP_K_PROPOSALS > 0:
            num_keep = min(pr_bboxes.shape[0], self.cfg.TEST_DEFAULT_TOP_K_PROPOSALS)
            pr_bboxes = pr_bboxes[:num_keep, :]
            pr_masks = pr_masks[:num_keep, :, :]
            assert pr_bboxes.shape[0] == pr_masks.shape[0], \
                '[ERROR] rois & masks size mismatch: {} vs. {}'.\
                format(pr_bboxes.shape[0], pr_masks.shape[0])
        
        # resize & build data blob
        TEST_SCALES = self.cfg.TEST_DEFAULT_SCALES
        _, im_scales = self.data_gen.built_image_blob(image, 
                                                       random_scale_id=None, 
                                                       phase='test',
                                                       scale_range=[0,len(TEST_SCALES)])
        orig_boxes = pr_bboxes.copy()
        #orig_masks = pr_masks.copy()
        orig_masks = or_masks_pooled
        
        # compute rois & masks blob over all scales
        pr_bboxes = self.data_gen.pred_rois_for_blob(pr_bboxes, im_scales, backbone_net)
        #pr_masks = self.data_gen.pred_masks_for_blob(pr_masks, im_scales, backbone_net)  
        
        GROUP_SCALE = self.cfg.TEST_DEFAULT_GROUP_SCALE
        MAX_ROIS_GPU = self.cfg.TEST_DEFAULT_MAX_ROIS_GPU
        # number of iter per group of scales
        num_scale_iter = int(np.ceil(len(TEST_SCALES) / float(GROUP_SCALE)))
        LO_SCALE = 0
        
        # set up return results
        final_boxes = np.zeros((0, 4), dtype=np.float32)
        final_masks = np.zeros((0, self.cfg.MAIN_DEFAULT_MASK_SIZE, 
                                   self.cfg.MAIN_DEFAULT_MASK_SIZE), dtype=np.float32)
        
        final_boxes_scores = np.zeros((0, self.data_gen.num_cls), dtype=np.float32)
        final_masks_scores = np.zeros((0, self.data_gen.num_cls), dtype=np.float32)
        
        for scale_iter in xrange(num_scale_iter):
            HI_SCALE = min(LO_SCALE + GROUP_SCALE, len(TEST_SCALES))
            inds_this_scale = np.where((pr_bboxes[:, 0] >= LO_SCALE) & (pr_bboxes[:, 0] < HI_SCALE))[0]
            if len(inds_this_scale) == 0:
                LO_SCALE += GROUP_SCALE
                continue
            
            max_rois_this_scale = MAX_ROIS_GPU[scale_iter]
            boxes_this_scale = pr_bboxes[inds_this_scale, :]
            masks_this_scale = pr_masks[inds_this_scale, :, :]
            num_iter_this_scale = int(np.ceil(boxes_this_scale.shape[0] / float(max_rois_this_scale)))
            # make the batch index of input box start from 0
            boxes_this_scale[:, 0] -= min(boxes_this_scale[:, 0])
            # re-prepare im blob for this_scale
            input_blobs = {}
            input_blobs['data'], _ = self.data_gen.built_image_blob(image, 
                                                       random_scale_id=None, 
                                                       phase='test',
                                                       scale_range=[LO_SCALE,HI_SCALE])
            input_blobs['data'] = input_blobs['data'].astype(np.float32, copy=False)
            input_start = 0
            for _ in xrange(num_iter_this_scale):
                input_end = min(input_start + max_rois_this_scale, boxes_this_scale.shape[0])
                input_box = boxes_this_scale[input_start:input_end, :]
                input_mask = masks_this_scale[input_start:input_end, :, :]
                input_blobs['rois'] = input_box.astype(np.float32, copy=False)
                # reshape (n_mask,h,w) -> (n_mask,1,h,w)
                input_mask = np.reshape(input_mask, (input_mask.shape[0],1,input_mask.shape[1],input_mask.shape[2]))
                #input_blobs['masks'] = input_mask
                input_blobs['masks'] = input_mask.astype(np.float32, copy=False)
                input_blobs['masks'] = (input_blobs['masks'] >= \
                    self.cfg.MAIN_DEFAULT_BINARIZE_THRESH).astype(np.float32, copy=False)
                self.net.blobs['data'].reshape(*input_blobs['data'].shape)
                self.net.blobs['rois'].reshape(*input_blobs['rois'].shape)
                self.net.blobs['masks'].reshape(*input_blobs['masks'].shape)
                #print 'input_blobs.keys(): {}'.format(input_blobs.keys())
                
                # perform a feed-forward
                blobs_out = self.net.forward(**input_blobs)
                
                output_bbox_prob = blobs_out['bbox_prob'].copy()
                output_mask_cls_prob = blobs_out['mask_cls_prob'].copy()                
                if cfm_t == 'T3':
                    output_mask_bit_prob = blobs_out['mask_bit_prob'].copy()
                    output_mask_bit_prob = np.reshape(output_mask_bit_prob, (output_mask_bit_prob.shape[0],
                                                                             final_masks.shape[1], 
                                                                             final_masks.shape[2]))
                    #print 'final_masks.shape: {}'.format(final_masks.shape)
                    print 'output_mask_bit_prob.shape: {}'.format(output_mask_bit_prob.shape)                    
                    final_masks = np.vstack((final_masks, output_mask_bit_prob))                    
                
                input_start += max_rois_this_scale
                final_boxes_scores = np.vstack((final_boxes_scores, output_bbox_prob))
                final_masks_scores = np.vstack((final_masks_scores, output_mask_cls_prob))            
            
            #print 'len(inds_this_scale): {}'.format(len(inds_this_scale))
            #print 'final_boxes.shape: {}'.format(final_boxes.shape)
            #print 'orig_boxes.shape: {}'.format(orig_boxes.shape)
            #print 'final_masks.shape: {}'.format(final_masks.shape)
            #print 'orig_masks.shape: {}'.format(orig_masks.shape)
            
            sel_orig_boxes = orig_boxes[inds_this_scale, :]
            final_boxes = np.vstack((final_boxes, sel_orig_boxes))
            if cfm_t == 'T1' or cfm_t == 'T2':
                sel_orig_masks = orig_masks[inds_this_scale, :]
                final_masks = np.vstack((final_masks, sel_orig_masks))
                
            LO_SCALE += GROUP_SCALE            
            #print 'len(final_boxes): {}'.format(len(final_boxes))
            #print 'len(final_masks): {}'.format(len(final_masks))
                        
            cfm_outputs = {'final_boxes': final_boxes,
                       'final_masks': final_masks,
                       'final_boxes_scores': final_boxes_scores,
                       'final_masks_scores': final_masks_scores
                       }           
            
        return  cfm_outputs
    
    def mnc_net_forward(self):
        pass
    
    
    def faster_rcnn_net_forward(self):
        pass
    
    def test_cfm_net(self, cfm_t, backbone_net):
        # detection threshold for each class
        # (this is adaptively set based on the max_per_set constraint)
        thresh = -np.inf * np.ones(self.data_gen.num_cls)
        # top_scores will hold one min heap of scores per class (used to enforce
        # the max_per_set constraint)
        top_scores = [[] for _ in xrange(self.data_gen.num_cls)]
        
        images = self.data_gen.images
        
        # all detections and segmentation are collected into a list:
        # Since the number of dets/segs are of variable size
        all_boxes = [[[] for _ in xrange(len(images))]
                     for _ in xrange(self.data_gen.num_cls)]
        all_masks = [[[] for _ in xrange(self.data_gen.num_cls)]
                     for _ in xrange(self.data_gen.num_cls)]
        _t = {'im_detect': Timer(), 'misc': Timer()}       
        
        for im_i in xrange(len(images)):
            _t['im_detect'].tic()
            image = images[im_i]
            
            cfm_outputs = self.cfm_net_forward(cfm_t, image, backbone_net)
            
            out_boxes = cfm_outputs['final_boxes']
            out_masks = cfm_outputs['final_masks']
            seg_scores = cfm_outputs['final_masks_scores']
            
            for j in xrange(1, self.data_gen.num_cls):
                inds = np.where(seg_scores[:, j] > thresh[j])[0]
                cls_scores = seg_scores[inds, j]
                cls_boxes = out_boxes[inds, :]
                cls_masks = out_masks[inds, :]
                top_inds = np.argsort(-cls_scores)[:self.max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                cls_masks = cls_masks[top_inds, :]
                # push new scores onto the min heap
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the min heap and update the class threshold
                if len(top_scores[j]) > self.max_per_set:
                    while len(top_scores[j]) > self.max_per_set:
                        heapq.heappop(top_scores[j])
                    thresh[j] = top_scores[j][0]
                box_before_nms = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
                    .astype(np.float32, copy=False)
                mask_before_nms = cls_masks.astype(np.float32, copy=False)
                all_boxes[j][im_i], all_masks[j][im_i] = apply_nms_mask_single(box_before_nms, 
                                                                               mask_before_nms,
                                                                               self.cfg.TEST_DEFAULT_NMS,
                                                                               self.cfg)
            _t['im_detect'].toc()
            print 'process image %d/%d, forward average time %f' % (im_i, len(self.data_gen.images),
                                                                    _t['im_detect'].average_time)
            
        for j in xrange(1, self.data_gen.num_cls):
            for i in xrange(len(images)):
                inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
                all_boxes[j][i] = all_boxes[j][i][inds, :]
                all_masks[j][i] = all_masks[j][i][inds]

        return all_boxes, all_masks
    
    def test_mnc_net(self):
        pass
    
    
    def test_model(self, gt_set, backbone_net, dataset, kwargs=None):
        output_dir = self.output_dir
        det_file = os.path.join(output_dir, dataset.name+'_final_boxes.pkl')
        seg_file = os.path.join(output_dir, dataset.name+'_final_masks.pkl')
        
        assert self.task in self.cfg.MAIN_DEFAULT_TASKS, \
            '[ERROR] unknown task name provided: {}'.format(self.task)
        
        if self.task == 'CFM':
            """ Testing segmentation using CFM """
            print '[INFO] Testing segmentation using CFM...'
            if os.path.isfile(det_file) and os.path.isfile(seg_file):
                with open(det_file, 'rb') as f:
                    final_boxes = cPickle.load(f)
                with open(seg_file, 'rb') as f:
                    final_masks = cPickle.load(f)
            else:
                assert 'cfm_t' in kwargs.keys(), \
                    '[ERROR] missing type of CFM to process'
                cfm_t = kwargs['cfm_t']
                final_boxes, final_masks = self.test_cfm_net(cfm_t, backbone_net)
                
                with open(det_file, 'wb') as f:
                    cPickle.dump(final_boxes, f, cPickle.HIGHEST_PROTOCOL)
                with open(seg_file, 'wb') as f:
                    cPickle.dump(final_masks, f, cPickle.HIGHEST_PROTOCOL)
                    
            """ Evaluating segmentation using CFM """
            print '[INFO] Evaluating segmentation using CFM...'
            pascal_eval = PascalEval(dataset=self.data_gen.dataset,
                                     task=self.task, cfg=self.cfg)
            pascal_eval.evaluate_segmentation(final_boxes, final_masks, gt_set, output_dir)
            
        elif self.task == 'MNC':
            raise NotImplemented
        
        elif self.task == 'FCIS':
            raise NotImplemented
        
        elif self.task == 'MRCNN':
            raise NotImplemented
        
        return final_boxes, final_masks
        
    
    
    
    
