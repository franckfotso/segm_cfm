# Project: segm_cfm
# Module: libs.metric.PascalEval
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Based on:
#    MNC: https://github.com/daijifeng001/MNC
# Licensed under MIT License

import os, cPickle, cv2
import numpy as np
import os.path as osp
import scipy.io as sio

from BasicEval import BasicEval
from utils.transformation import mask_overlap

class PascalEval(BasicEval):
    
    def __init__(self,
                 dataset,
                 task, cfg):
        
        super(PascalEval, self).__init__(dataset, task, cfg)    
    
    def evaluate_segmentation(self, all_boxes, all_masks, gt_set, output_dir):
        im_names = self.dataset.sets[gt_set]["im_names"]        
        self._write_voc_seg_results_file(all_boxes, all_masks, output_dir)
        self._py_evaluate_segmentation(im_names, output_dir)

    def _write_voc_seg_results_file(self, all_boxes, all_masks, output_dir):
        """
        Write results as a pkl file, note this is different from
        detection task since it's difficult to write masks to txt
        """
        # Always reformat result in case of sometimes masks are not
        # binary or is in shape (n, sz*sz) instead of (n, sz, sz)
        all_boxes, all_masks = self._reformat_result(all_boxes, all_masks)
        for cls_inds, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = os.path.join(output_dir, cls + '_det.pkl')
            with open(filename, 'wr') as f:
                cPickle.dump(all_boxes[cls_inds], f, cPickle.HIGHEST_PROTOCOL)
            filename = os.path.join(output_dir, cls + '_seg.pkl')
            with open(filename, 'wr') as f:
                cPickle.dump(all_masks[cls_inds], f, cPickle.HIGHEST_PROTOCOL)

    def _reformat_result(self, boxes, masks):
        num_images = len(self.image_index)
        num_class = len(self.classes)
        reformat_masks = [[[] for _ in xrange(num_images)]
                          for _ in xrange(num_class)]
        for cls_inds in xrange(1, num_class):
            for img_inds in xrange(num_images):
                if len(masks[cls_inds][img_inds]) == 0:
                    continue
                num_inst = masks[cls_inds][img_inds].shape[0]
                reformat_masks[cls_inds][img_inds] = masks[cls_inds][img_inds]\
                    .reshape(num_inst, self.cfg.MAIN_DEFAULT_MASK_SIZE, 
                                       self.cfg.MAIN_DEFAULT_MASK_SIZE)
                reformat_masks[cls_inds][img_inds] = \
                    reformat_masks[cls_inds][img_inds] >= self.cfg.MAIN_DEFAULT_BINARIZE_THRESH
        all_masks = reformat_masks
        return boxes, all_masks

    def _py_evaluate_segmentation(self, im_names, in_gt_dir, output_dir):
        cache_dir = os.path.join(self.cfg.MAIN_DIR_ROOT, 
                                 self.cfg.MAIN_DIR_CACHE, 'eval_annotations')
        aps = []
        # define this as true according to SDS's evaluation protocol
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        print '~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~'
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = self.voc_eval_sds(det_filename, seg_filename, in_gt_dir,
                              im_names, cls, cache_dir, self._classes, ov_thresh=0.5)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
        print('Mean AP@0.5 = {:.2f}'.format(np.mean(aps)*100))
        print '~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~'
        aps = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = self.voc_eval_sds(det_filename, seg_filename, in_gt_dir,
                              im_names, cls, cache_dir, self._classes, ov_thresh=0.7)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
        print('Mean AP@0.7 = {:.2f}'.format(np.mean(aps)*100))
        
    def voc_eval_sds(self, det_file, seg_file, in_gt_dir, im_names, cls_name, cache_dir,
                 class_names, ov_thresh=0.5):
        # 1. Check whether ground truth cache file exists
        self.check_voc_sds_cache(cache_dir, in_gt_dir, im_names, class_names)
        gt_cache = cache_dir + '/' + cls_name + '_mask_gt.pkl'
        with open(gt_cache, 'rb') as f:
            gt_pkl = cPickle.load(f)
    
        # 2. Get predict pickle file for this class
        with open(det_file, 'rb') as f:
            boxes_pkl = cPickle.load(f)
        with open(seg_file, 'rb') as f:
            masks_pkl = cPickle.load(f)
    
        # 3. Pre-compute number of total instances to allocate memory
        num_image = len(im_names)
        box_num = 0
        for im_i in xrange(num_image):
            box_num += len(boxes_pkl[im_i])
    
        # 4. Re-organize all the predicted boxes
        new_boxes = np.zeros((box_num, 5))
        new_masks = np.zeros((box_num, self.cfg.MAIN_DEFAULT_MASK_SIZE, 
                                       self.cfg.MAIN_DEFAULT_MASK_SIZE))
        new_image = []
        cnt = 0
        for image_ind in xrange(len(im_names)):
            boxes = boxes_pkl[image_ind]
            masks = masks_pkl[image_ind]
            num_instance = len(boxes)
            for box_ind in xrange(num_instance):
                new_boxes[cnt] = boxes[box_ind]
                new_masks[cnt] = masks[box_ind]
                new_image.append(im_names[image_ind])
                cnt += 1
    
        # 5. Rearrange boxes according to their scores
        seg_scores = new_boxes[:, -1]
        keep_inds = np.argsort(-seg_scores)
        new_boxes = new_boxes[keep_inds, :]
        new_masks = new_masks[keep_inds, :, :]
        num_pred = new_boxes.shape[0]
    
        # 6. Calculate t/f positive
        fp = np.zeros((num_pred, 1))
        tp = np.zeros((num_pred, 1))
        for i in xrange(num_pred):
            pred_box = np.round(new_boxes[i, :4]).astype(int)
            pred_mask = new_masks[i]
            pred_mask = cv2.resize(pred_mask.astype(np.float32), (pred_box[2] - pred_box[0] + 1, pred_box[3] - pred_box[1] + 1))
            pred_mask = pred_mask >= self.cfg.MAIN_DEFAULT_BINARIZE_THRESH
            image_index = new_image[keep_inds[i]]
    
            if image_index not in gt_pkl:
                fp[i] = 1
                continue
            gt_dict_list = gt_pkl[image_index]
            # calculate max region overlap
            cur_overlap = -1000
            cur_overlap_ind = -1
            for ind2, gt_dict in enumerate(gt_dict_list):
                gt_mask_bound = np.round(gt_dict['mask_bound']).astype(int)
                pred_mask_bound = pred_box
                ov = mask_overlap(gt_mask_bound, pred_mask_bound, gt_dict['mask'], pred_mask)
                if ov > cur_overlap:
                    cur_overlap = ov
                    cur_overlap_ind = ind2
            if cur_overlap >= ov_thresh:
                if gt_dict_list[cur_overlap_ind]['already_detect']:
                    fp[i] = 1
                else:
                    tp[i] = 1
                    gt_dict_list[cur_overlap_ind]['already_detect'] = 1
            else:
                fp[i] = 1
    
        # 7. Calculate precision
        num_pos = 0
        for key, val in gt_pkl.iteritems():
            num_pos += len(val)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(num_pos)
        # avoid divide by zero in case the first matches a difficult gt
        prec = tp / np.maximum(fp+tp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, True)
        
        return ap
    
    
    def check_voc_sds_cache(self, cache_dir, in_gt_dir, im_names, class_names):
        """
        Args:
            cache_dir: output directory for cached mask annotation
            devkit_path: root directory of VOCdevkitSDS
            image_names: used for parse image instances
            class_names: VOC 20 class names
        """
    
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
    
        exist_cache = True
        for cls_name in class_names:
            if cls_name == '__background__':
                continue
            cache_name = os.path.join(cache_dir, cls_name + '_mask_gt.pkl')
            if not os.path.isfile(cache_name):
                exist_cache = False
                break
    
        if not exist_cache:
            # load annotations:
            # create a list with size classes
            record_list = [{} for _ in xrange(21)]
            for i, image_name in enumerate(im_names):
                record = self.parse_inst(image_name, in_gt_dir)
                for j, mask_dic in enumerate(record):
                    cls = mask_dic['mask_cls']
                    mask_dic['already_detect'] = False
                    if image_name not in record_list[cls]:
                        record_list[cls][image_name] = []
                    record_list[cls][image_name].append(mask_dic)
                if i % 100 == 0:
                    print 'Reading annotation for {:d}/{:d}'.format(i + 1, len(im_names))
    
            print 'Saving cached annotations...'
            for cls_ind, name in enumerate(class_names):
                if name == '__background__':
                    continue
                cachefile = os.path.join(cache_dir, name + '_mask_gt.pkl')
                with open(cachefile, 'w') as f:
                    cPickle.dump(record_list[cls_ind], f)
    
    
    def parse_inst(self, im_name, gt_mat_dir):
        """
        Get cooresponding masks, boxes, classes according to image name
        Args:
            im_name: input image name
            gt_mat_dir: root dir gt mat images
        Returns:
            roi/mask dictionary of this image
        """
        gt_mat_pn = osp.join(gt_mat_dir, im_name+'.mat')
        assert osp.exists(gt_mat_pn), \
        '[ERROR] missing mat file: {}'.format(gt_mat_pn)            
        gt_mat = sio.loadmat(gt_mat_pn)
        
        gt_labels = gt_mat['labels'][0]
        inst_ids = gt_mat['inst_ids'][0]
        gt_superpixels = gt_mat['superpixels']
        gt_bboxes = gt_mat['bboxes']
        
        record = []
        for m_i, (label, inst_id) in enumerate(zip(gt_labels, inst_ids)):
            [row_ids, col_ids] = np.array(np.where((gt_superpixels == inst_id)))
            mask = np.zeros(gt_superpixels.shape, dtype=np.float)
            mask[row_ids, col_ids] = 1
            mask_bound = gt_bboxes[m_i]
            
            record.append({
                'mask': mask,
                'mask_cls': label,
                'mask_bound': mask_bound
            })
    
        return record
    
    