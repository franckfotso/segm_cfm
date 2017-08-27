# Project: segm_cfm
# Module: libs.datasets.PascalGenerator
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Based on: py-faster-rcnn 
#    (https://github.com/rbgirshick/py-faster-rcnn)
# Licensed under MIT License

import scipy.io as sio
import numpy.random as npr
from datasets.IMGenerator import IMGenerator
from utils.regression import *
from utils.transformation import *

class CFMGenerator(IMGenerator):
        
    def __init__(self, images, masks, num_cls, 
                 cfm_t, bbox_means, bbox_stds, 
                 in_gt_dir, in_pr_dir, cache_im_dir, cfg):
        self._images = images
        self._masks = masks       
        self._num_cls = num_cls
        self._cfm_t = cfm_t
        self._bbox_means = bbox_means
        self._bbox_stds = bbox_stds
        self._in_gt_dir = in_gt_dir
        self._in_pr_dir = in_pr_dir
        self._cache_im_dir = cache_im_dir
        self._cfg = cfg
        
        self._cur_idx, self._perm_ids = self.shuffe_images()
                
        super(CFMGenerator, self).__init__()

    def get_in_gt_dir(self):
        return self._in_gt_dir


    def get_in_pr_dir(self):
        return self._in_pr_dir


    def set_in_gt_dir(self, value):
        self._in_gt_dir = value


    def set_in_pr_dir(self, value):
        self._in_pr_dir = value


    def del_in_gt_dir(self):
        del self._in_gt_dir


    def del_in_pr_dir(self):
        del self._in_pr_dir


    def get_cache_im_dir(self):
        return self._cache_im_dir


    def set_cache_im_dir(self, value):
        self._cache_im_dir = value


    def del_cache_im_dir(self):
        del self._cache_im_dir


    def get_bbox_means(self):
        return self._bbox_means


    def get_bbox_stds(self):
        return self._bbox_stds


    def set_bbox_means(self, value):
        self._bbox_means = value


    def set_bbox_stds(self, value):
        self._bbox_stds = value


    def del_bbox_means(self):
        del self._bbox_means


    def del_bbox_stds(self):
        del self._bbox_stds


    def get_num_cls(self):
        return self._num_cls


    def set_num_cls(self, value):
        self._num_cls = value


    def del_num_cls(self):
        del self._num_cls


    def get_masks(self):
        return self._masks


    def set_masks(self, value):
        self._masks = value


    def del_masks(self):
        del self._masks


    def get_perm_ids(self):
        return self._perm_ids


    def get_images(self):
        return self._images


    def set_perm_ids(self, value):
        self._perm_ids = value


    def set_images(self, value):
        self._images = value


    def del_perm_ids(self):
        del self._perm_ids


    def del_images(self):
        del self._images


    def get_cur_idx(self):
        return self._cur_idx


    def set_cur_idx(self, value):
        self._cur_idx = value


    def del_cur_idx(self):
        del self._cur_idx


    def get_cfg(self):
        return self._cfg


    def set_cfg(self, value):
        self._cfg = value


    def del_cfg(self):
        del self._cfg

    
    def shuffe_images(self):
        """Randomly permute the training images"""        
        
        if self.cfg.TRAIN_DEFAULT_ASPECT_GROUPING:
            widths = np.array([im.pr_rois['im_info']['width'] for im in self.images])            
            heights = np.array([im.pr_rois['im_info']['height'] for im in self.images])
            
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            self.perm_ids = np.reshape(inds[row_perm, :], (-1,))
        else:
            self.perm_ids = np.random.permutation(np.arange(len(self.images)))    
        
        self.cur_idx = 0
        return self.cur_idx, self.perm_ids
    
    
    def get_next_minibatch_ids(self):
        
        if self.cur_idx + self.cfg.TRAIN_BATCH_NUM_IMG >= len(self.images):
            self.shuffe_images()
            
        batch_ids = self.perm_ids[self.cur_idx:self.cur_idx+self.cfg.TRAIN_BATCH_NUM_IMG]
        self.cur_idx += self.cfg.TRAIN_BATCH_NUM_IMG
        
        return batch_ids
    
    
    def get_next_minibatch(self):
        
        batch_ids = self.get_next_minibatch_ids()
        minibatch_imgs = [self.images[i] for i in batch_ids]
        minibatch_masks = [self.masks[i] for i in batch_ids]
        
        return self.get_minibatch(minibatch_imgs, minibatch_masks, self.cache_im_dir)
        
        
    def get_minibatch(self, minibatch_imgs, minibatch_masks, cache_im_dir):
        """Given an image obj, construct a minibatch sampled from it."""
        
        assert len(minibatch_imgs) == 1, \
            "[ERROR] Single batch only, found: {}".format(len(minibatch_imgs))
        assert len(minibatch_masks) == 1, \
            "[ERROR] Single batch only, found: {}".format(len(minibatch_imgs))
        
        """ load data from cache (precomputed data: blobs) """
        
        num_imgs = len(minibatch_imgs)
        rois_blob = np.zeros((0, 5), dtype=np.float32)   
        bbox_labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * self.num_cls), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        
        masks_blob = None
        mask_labels_blob = np.zeros((0), dtype=np.float32)
        mask_targets_blob = np.zeros((0, 0, self.cfg.MAIN_DEFAULT_MASK_SIZE, 
                                      self.cfg.MAIN_DEFAULT_MASK_SIZE), dtype=np.float32)
        mask_weights_blob = np.zeros((0, 0, self.cfg.MAIN_DEFAULT_MASK_SIZE, 
                                      self.cfg.MAIN_DEFAULT_MASK_SIZE), dtype=np.float32)                  
        
        for im_i in xrange(num_imgs):
            image = minibatch_imgs[im_i]
            #print 'get_minibatch, image: {}'.format(image.filename)
              
            im_nm = (image.filename).split('.')[0]
            cache_im_pn = osp.join(cache_im_dir, im_nm+'.mat')
            
            assert osp.exists(cache_im_pn), \
            '[ERROR] missing precomputed data for the image: {}'.format(cache_im_pn)              
            cache_im_data = sio.loadmat(cache_im_pn)
            
            try:
                """ 1. Get precomputed data from cache """
                im_blob = cache_im_data['im_blob']            
                im_scales = cache_im_data['im_scales'][0]          
                sample_rois = cache_im_data['sample_rois']
                sample_masks = cache_im_data['sample_masks']         
                """--------------- ROIs ---------------"""
                fg_inds = sample_rois['fg_inds'][0,0][0]
                gt_assignment = sample_rois['roi_gt_assignment'][0,0][0]
                sel_rois = sample_rois['sel_rois'][0,0]
                # (0,x1,y1,x2,y2) -> (x1,y1,x2,y2)
                sel_rois = sel_rois[:,1:5]
                bbox_labels = sample_rois['bbox_labels'][0,0][0]
                bbox_targets = sample_rois['bbox_targets'][0,0]
                bbox_inside_weights = sample_rois['bbox_inside_weights'][0,0]
            except Exception as e:
                print ("[ERROR] exception found: {}, on file: {}".format(e, cache_im_pn))
                #raise Exception('[ERROR] failed to handle image: {}'.format(cache_im_pn))
            
            # Apply means & stds to bbox_targets
            """
            print 'bbox_labels.shape: {}'.format(bbox_labels.shape)
            print 'bbox_labels: {}'.format(bbox_labels)
            print 'gt_assignment: {}'.format(gt_assignment)
            print 'fg_inds: {}'.format(fg_inds)
            #"""
            if self.cfg.TRAIN_DEFAULT_BBOX_REG:
                for i in xrange(len(fg_inds)):
                    cls = bbox_labels[gt_assignment[fg_inds[i]]]
                    if cls == 0:
                        continue
                    bbox_targets[i, cls*4:cls*4+4] -= self.bbox_means[cls, :]
                    bbox_targets[i, cls*4:cls*4+4] /= self.bbox_stds[cls, :]            
            
            """--------------- Masks ---------------"""
            sel_masks = sample_masks['sel_masks'][0,0]
            sel_masks = (sel_masks >= self.cfg.MAIN_DEFAULT_BINARIZE_THRESH).astype(bool)
            mask_labels = sample_masks['mask_labels'][0,0][0]
            mask_targets = sample_masks['mask_targets'][0,0]
            mask_targets = (mask_targets >= self.cfg.MAIN_DEFAULT_BINARIZE_THRESH).astype(bool)
            # (num_masks,h,w) -> (num_im,num_masks,h,w)
            #mask_targets = np.reshape(mask_targets, ((1,)+mask_targets.shape))
            mask_target_weights = sample_masks['mask_target_weights'][0,0]
            
            """ 2. Add to RoIs blob """           
            sel_rois = self.project_im_rois(sel_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((sel_rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, sel_rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            """ 3. Add to bbox_labels, bbox_targets, and bbox_inside_weights """
            bbox_labels_blob = np.hstack((bbox_labels_blob, bbox_labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            bbox_outside_blob = np.array(bbox_inside_blob > 0).astype(np.float32)           
            
            """ 4. setup masks blob """
            sel_masks = np.reshape(sel_masks, (1,)+sel_masks.shape) # => n_im, n_pr, h, w
            masks_blob = self.masks_to_blob(sel_masks)
            
            """ 5. setup mask_labels, mask_targets, mask_weights blob """            
            mask_labels_blob = np.hstack((mask_labels_blob, mask_labels))
            mask_targets_blob = mask_targets # TODO: update shape for img batch > 1
            mask_weights_blob = mask_target_weights # TODO: update shape for img batch > 1 
    
        if self.cfm_t == 'T1':
            blobs = {
                 'data': im_blob, 
                 'rois': rois_blob,
                 'masks': masks_blob,
                 'bbox_labels': bbox_labels_blob,
                 'mask_labels': mask_labels_blob
                 }
              
        elif self.cfm_t == 'T2':
            blobs = {
                 'data': im_blob, 
                 'rois': rois_blob,
                 'masks': masks_blob,
                 'bbox_labels': bbox_labels_blob,
                 'mask_labels': mask_labels_blob,
                 'bbox_targets': bbox_targets_blob,
                 'bbox_inside_weights': bbox_inside_blob,
                 'bbox_outside_weights': bbox_outside_blob}
            
        elif self.cfm_t == 'T3' or self.cfm_t == 'T4':
            blobs = {
                 'data': im_blob, 
                 'rois': rois_blob,
                 'masks': masks_blob,
                 'bbox_labels': bbox_labels_blob,
                 'mask_labels': mask_labels_blob,
                 'bbox_targets': bbox_targets_blob,
                 'mask_targets': mask_targets_blob,
                 'bbox_inside_weights': bbox_inside_blob,
                 'bbox_outside_weights': bbox_outside_blob,
                 'mask_weights': mask_weights_blob
                 }
        else:
            raise Exception('[ERROR] unknow CFM type')
        
        return blobs
        
    def built_image_blob(self, image, random_scale_id, phase='train', scale_range=None):
        im_RAW = cv2.imread(image.pathname)        
        if image.pr_rois['flipped']:
            im_RAW = im_RAW[:, ::-1, :]
            
        built_im_RAWs = []
        built_im_scales = []
            
        if phase == 'train':            
            target_size = self.cfg.TRAIN_DEFAULT_SCALES[random_scale_id]
            PIXEL_MEANS = np.array([[self.cfg.MAIN_DEFAULT_PIXEL_MEANS]])
            im_RAW, im_scale = self.prep_im_for_blob(im_RAW, 
                                        PIXEL_MEANS, target_size,
                                        self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            built_im_RAWs.append(im_RAW)
            built_im_scales.append(im_scale)
            
        elif phase == 'test':
            for target_size in self.cfg.TEST_DEFAULT_SCALES[scale_range[0]:scale_range[1]]:
                PIXEL_MEANS = np.array([[self.cfg.MAIN_DEFAULT_PIXEL_MEANS]])
                im_RAW, im_scale = self.prep_im_for_blob(im_RAW, 
                                            PIXEL_MEANS, target_size,
                                            self.cfg.TRAIN_DEFAULT_MAX_SIZE)
                built_im_RAWs.append(im_RAW)
                built_im_scales.append(im_scale)                
        else:
            raise Exception('[ERROR] unknown phase: {}'.format(phase))
            
        
        # Create a blob to hold the input images
        blob = self.images_to_blob(built_im_RAWs)

        return blob, built_im_scales
    
    def built_all_masks(self, mask, random_scale):
        all_mask_RAWs = []
        all_mask_scales = []
        
        num_gt_mask_insts = len(mask.gt_data['instances'])
        num_pr_mask_insts = len(mask.pr_data['instances'])
        
        #u_gt_mask = np.unique(mask.gt_data['instances'][0])
        #print 'mask.gt_data.instances[0], u_gt_mask: {}'.format(u_gt_mask)
    
        """ build gt masks"""
        for m_i in xrange(num_gt_mask_insts):
            mask_inst_RAW = mask.gt_data['instances'][m_i]
                
            target_size = self.cfg.TRAIN_DEFAULT_SCALES[random_scale]
            mask_inst_RAW, mask_inst_scale = self.prep_mask_for_blob(mask_inst_RAW, 
                                        target_size, self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            
            #mask_inst_RAW = mask_inst_RAW > 1
            all_mask_RAWs.append(mask_inst_RAW)
            all_mask_scales.append(mask_inst_scale)
        
        #u_gt_mask = np.unique(all_mask_RAWs[0])
        #print 'all_mask_RAWs[0], u_gt_mask: {}'.format(u_gt_mask)
        
        """ build pr masks """
        for m_i in xrange(num_pr_mask_insts):
            mask_inst_RAW = mask.pr_data['instances'][m_i]
            
            target_size = self.cfg.TRAIN_DEFAULT_SCALES[random_scale]
            mask_inst_RAW, mask_inst_scale = self.prep_mask_for_blob(mask_inst_RAW, 
                                        target_size, self.cfg.TRAIN_DEFAULT_MAX_SIZE)
                
            all_mask_RAWs.append(mask_inst_RAW)
            all_mask_scales.append(mask_inst_scale) 
                                        
        return np.array(all_mask_RAWs), np.array(all_mask_scales)
    
    def prep_im_for_blob(self, im_RAW, pixel_means, target_size, max_size):
        
        im_RAW = im_RAW.astype(np.float32, copy=False)
        im_RAW -= pixel_means
        im_shape = im_RAW.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        
        im_scale = float(target_size) / float(im_size_min)        
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
            
        im_RAW = cv2.resize(im_RAW, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        
        return im_RAW, im_scale
    
    def pred_rois_for_blob(self, im_rois, im_scales, backbone_net=None):
        """
        Convert rois to network input
        support multi-scale testing
        """
        im_rois = im_rois.astype(np.float, copy=False)
        if len(im_scales) > 1:
            widths = im_rois[:, 2] - im_rois[:, 0] + 1
            heights = im_rois[:, 3] - im_rois[:, 1] + 1
    
            areas = widths * heights
            scaled_areas = areas[:, np.newaxis] * (im_scales[np.newaxis, :] ** 2)
            
            if backbone_net == 'AlexNet':
                im_input_shape = [227, 227]
            elif backbone_net == 'VGG16':
                im_input_shape = [224, 224]
            else:
                raise Exception('[ERROR] unknown BACKBONE_NET {}'.format(backbone_net))
                 
            diff_areas = np.abs(scaled_areas - (im_input_shape[0]*im_input_shape[1]))
            levels = diff_areas.argmin(axis=1)[:, np.newaxis]
        else:
            levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
            
        im_scales = np.array(im_scales)
        #print 'im_rois.shape: {}'.format(im_rois.shape)
        #print 'levels.shape: {}'.format(levels.shape)
        #print 'im_scales.shape: {}'.format(im_scales.shape)
        im_rois = im_rois * im_scales[levels]
        rois_blob = np.hstack((levels.astype(np.float), im_rois))
        return rois_blob
    
    def pred_masks_for_blob(self, masks, im_scales, backbone_net=None):
        """
            Convert mask to network input
            support multi-scale testing
        """
        num_masks = masks.shape[0]
        if len(im_scales) > 1:
            heights = masks.shape[3] + 1
            widths = masks.shape[4] + 1           
    
            areas = widths * heights
            scaled_areas = areas[:, np.newaxis] * (im_scales[np.newaxis, :] ** 2)
            
            if backbone_net == 'AlexNet':
                im_input_shape = [227, 227]
            elif backbone_net == 'VGG16':
                im_input_shape = [224, 224]
            else:
                raise Exception('[ERROR] unknown BACKBONE_NET {}'.format(backbone_net))
                 
            diff_areas = np.abs(scaled_areas - (im_input_shape[0]*im_input_shape[1]))
            levels = diff_areas.argmin(axis=1)[:, np.newaxis]
        else:
            levels = np.zeros((num_masks, 1), dtype=np.int)
            
        im_scales = np.array(im_scales)
        masks_scales = im_scales[levels]
        #print 'masks.shape: {}'.format(masks.shape)
        masks = np.reshape(masks, (num_masks, masks.shape[1], masks.shape[2]))
        
        assert num_masks == len(masks_scales), \
            '[ERROR] num_masks != number of scales, {} vs. {}'.format(num_masks, len(masks_scales))
        
        masks_scaled = []
        for m_i in xrange(num_masks):
            mask = cv2.resize(masks[m_i], None, None,
                                fx=masks_scales[m_i], fy=masks_scales[m_i],
                                interpolation=cv2.INTER_LINEAR)
            masks_scaled.append(mask)
            
        masks_scaled = np.array(masks_scaled)        
        del masks
        
        return masks_scaled
        
    def prep_mask_for_blob(self, mask_RAW, target_size, max_size):
        mask_shape = mask_RAW.shape
        mask_size_min = np.min(mask_shape[0:2])
        mask_size_max = np.max(mask_shape[0:2])
        
        mask_scale = float(target_size) / float(mask_size_min)        
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(mask_scale * mask_size_max) > max_size:
            mask_scale = float(max_size) / float(mask_size_max)
                
        mask_RAW = (cv2.resize(mask_RAW, None, None, fx=mask_scale, fy=mask_scale,
                        interpolation=cv2.INTER_LINEAR)).astype(np.int)
        
        
        return mask_RAW, mask_scale
    
    def images_to_blob(self, im_RAWs):
        max_shape = np.array([im_RAW.shape for im_RAW in im_RAWs]).max(axis=0)
        num_images = len(im_RAWs)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
        
        for i in xrange(num_images):
            im_RAW = im_RAWs[i]
            blob[i, 0:im_RAW.shape[0], 0:im_RAW.shape[1], :] = im_RAW
    
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        
        return blob
    
    def masks_to_blob(self, all_mask_RAWs):
        num_masks = len(all_mask_RAWs)
        num_mask_insts_max = np.max([len(c) for c in all_mask_RAWs])
        
        shapes = []
        for all_mask_inst_RAWs in all_mask_RAWs:
            for mask_inst_RAW in all_mask_inst_RAWs:
                shapes.append(mask_inst_RAW.shape)                
        max_shape = np.array(shapes).max(axis=0)
        
        blob = np.zeros((num_masks, num_mask_insts_max, 
                         max_shape[0], max_shape[1],1),
                         dtype=np.float32)
        
        for m_i in xrange(num_masks):
            for m_j in xrange(len(all_mask_RAWs[m_i])):
                mask_inst_RAW = all_mask_RAWs[m_i][m_j]
                new_shape = (mask_inst_RAW.shape[0],mask_inst_RAW.shape[1],1)
                mask_inst_RAW = np.reshape(mask_inst_RAW, new_shape)
                blob[m_i,m_j,0:mask_inst_RAW.shape[0],0:mask_inst_RAW.shape[1],:] = mask_inst_RAW
        
        blob = np.reshape(blob, (num_mask_insts_max, max_shape[0], max_shape[1],1))
        # Move channels (axis 4) to axis 2
        # Axis order will become: (batch elem, mask_inst, channel, height, width)
        #channel_swap = (0, 1, 4, 2, 3)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        
        return blob
    
    
    def sample_rois_masks(self, all_rois, gt_boxes, num_classes, image, all_mask_RAWs, im_scales, cfg):
        """
            Generate a random sample of RoIs & Masks comprising 
            foreground and background examples.
        """
        num_imgs = 1
        assert(cfg.TRAIN_BATCH_NUM_SEGM % num_imgs == 0), \
                    'num_images ({}) must divide BATCH_SIZE ({})'. \
                    format(num_imgs, cfg.TRAIN_BATCH_NUM_SEGM)
                
        rois_per_image = cfg.TRAIN_BATCH_NUM_SEGM / num_imgs
        
        num_gt_mask = len(image.gt_rois['boxes'])
        num_all_rois = all_rois.shape[0]
        num_all_masks = all_mask_RAWs.shape[0]
        assert num_all_rois == num_all_masks, \
        "[ERROR] num_all_rois != num_all_masks: {} != {}".format(num_all_rois, num_all_masks)        
        
        """ compute rois & masks overlaps """
        # overlaps: (rois x gt_boxes)       
        roi_overlaps_all = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        roi_gt_assignment = roi_overlaps_all.argmax(axis=1) # rois -> gt_cls: num_rois*num_cls
        roi_overlaps_max = roi_overlaps_all.max(axis=1) # rois -> max_overlap_value (rel. gt_cls)
                
        # overlaps: (masks x gt_boxes)
        seg_overlaps_all = roi_overlaps_all
        seg_overlaps_max = np.zeros((num_all_masks, 1))        
        seg_gt_assignment = roi_gt_assignment
        seg_gt_assignment[roi_overlaps_max == 0] = -1
        # record mask region overlaps
        seg_overlaps_max[:num_gt_mask] = 1.0
        #print 'seg_assignment: {}'.format(seg_assignment)
        
        """ Sampling Masks """
        gt_mask_RAWs = all_mask_RAWs[0:num_gt_mask]
        for m_i in xrange(num_gt_mask, num_all_masks):
            pr_roi = all_rois[m_i][1:5]            
            # compute overlaps on gt_masks
            for m_j in xrange(num_gt_mask):
                gt_mask = gt_mask_RAWs[m_j]
                gt_roi = all_rois[m_j][1:5]
                gt_cls = all_rois[m_j][0]
                                
                crop_ov = mask_overlap(pr_roi, gt_roi, all_mask_RAWs[m_i], all_mask_RAWs[m_j])
                seg_overlaps_max[m_i] = max(seg_overlaps_max[m_i], crop_ov)
                #seg_overlaps_all[m_i][gt_cls] = max(seg_overlaps_all[m_i][gt_cls], crop_ov)
               
        #seg_gt_assignment = seg_overlaps_all.argmax(axis=1) # masks -> gt_cls: num_masks x num_cls
        #seg_overlaps_max = seg_overlaps_all.max(axis=1) # masks -> max_overlap_value (rel. gt_cls)
       
        """ Filtering rois & masks indexes """
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds_roi = np.where(roi_overlaps_max >= cfg.TRAIN_DEFAULT_ROI_FG_THRESH)[0]
        fg_inds_seg = np.where(seg_overlaps_max >= cfg.TRAIN_DEFAULT_SEG_FG_THRESH)[0]
        bg_inds_seg = np.where(seg_overlaps_max < cfg.TRAIN_DEFAULT_SEG_FG_THRESH)[0]

        keep_inds = []
        for i in xrange(len(cfg.TRAIN_BATCH_FRACTION_SAMPLE)):
            cur_keep_inds = np.where((roi_overlaps_max >= cfg.TRAIN_BATCH_THRESH_LO_SAMPLE[i]) &
                                     (roi_overlaps_max <= cfg.TRAIN_BATCH_THRESH_HI_SAMPLE[i]))[0]
            cur_rois_this_image = np.round(rois_per_image * cfg.TRAIN_BATCH_FRACTION_SAMPLE[i])
            cur_rois_this_image = int(min(cur_rois_this_image, len(cur_keep_inds)))
            if cur_keep_inds.size > 0:
                #print cur_rois_this_image
                cur_keep_inds = npr.choice(cur_keep_inds, size=cur_rois_this_image, replace=False)

            if i == 0:
                keep_inds = cur_keep_inds
            else:
                keep_inds = np.unique(np.hstack((keep_inds, cur_keep_inds)))

        fg_inds_roi = keep_inds[np.in1d(keep_inds, fg_inds_roi)]
        bg_inds_roi = keep_inds[np.in1d(keep_inds, fg_inds_roi, invert=True)]
        keep_inds = np.append(fg_inds_roi, bg_inds_roi).astype(int)
        
        """ resize masks for mask_pooling layer"""
        sel_masks = all_mask_RAWs[keep_inds]
        sel_rois = all_rois[keep_inds]
        mask_shape = tuple(cfg.MAIN_DEFAULT_MASK_POOL_SHAPE)
        all_mask_pooled = []
        for m_i in xrange(len(sel_masks)):
            box = sel_rois[m_i][1:5]
            mask_RAW = sel_masks[m_i]
            
            mask_RAW_crop = mask_RAW[box[1]:box[3]+1,box[0]:box[2]+1]
            mask_pooled = cv2.resize(mask_RAW_crop.astype(np.float32), mask_shape)
            all_mask_pooled.append(mask_pooled)
        all_mask_pooled = np.array(all_mask_pooled)
        del all_mask_RAWs        
        sel_masks = all_mask_pooled
        
        """ Sampling ROIs """
        # Select sampled values from various arrays:
        bbox_labels = gt_boxes[seg_gt_assignment, 4]
        bbox_labels = bbox_labels[keep_inds]
        # Clamp labels for the background RoIs to 0
        bbox_labels[len(fg_inds_roi):] = 0
        roi_overlaps_all = roi_overlaps_all[keep_inds]
        roi_overlaps_max = roi_overlaps_max[keep_inds]
        #print 'keep_inds: {}'.format(keep_inds)
        #print 'sel_rois[0:6]: {}'.format(sel_rois[0:6])
        
        # project rois to image's scale bef. compute targets
        assert len(im_scales) == 1, \
            "[ERROR]: only one scale value required"
            
        ex_rois = sel_rois[:, 1:5] * im_scales[0]
        gt_rois = gt_boxes[seg_gt_assignment[keep_inds], :4] * im_scales[0]        
        bbox_target_data = compute_bbox_targets(ex_rois, gt_rois, bbox_labels, False, cfg)
            
        bbox_targets, bbox_inside_weights = \
            get_bbox_regression_labels(bbox_target_data, num_classes, cfg)
            
        #print 'sel_rois.shape: {}'.format(sel_rois.shape)        
        sample_rois = { 'fg_inds': fg_inds_roi,
                        'roi_overlaps_max': roi_overlaps_max,
                        'roi_overlaps_all': roi_overlaps_all,
                        'roi_gt_assignment': roi_gt_assignment,
                        'sel_rois': sel_rois,
                        'bbox_labels': bbox_labels,
                        'bbox_targets': bbox_targets,
                        'bbox_target_data': bbox_target_data,
                        'bbox_inside_weights': bbox_inside_weights}
            
        """ Sampling Masks """
        mask_tgt_shape = (self.cfg.MAIN_DEFAULT_MASK_SIZE, self.cfg.MAIN_DEFAULT_MASK_SIZE)
        mask_targets = np.zeros((num_all_masks,)+mask_tgt_shape)
        
        for m_i in xrange(num_all_masks):
            if seg_gt_assignment[m_i] == -1:
                continue
            sel_idx = seg_gt_assignment[m_i]
            pr_roi = all_rois[m_i, 1:5]
            gt_roi = all_rois[sel_idx][1:5]            
            gt_mask = gt_mask_RAWs[sel_idx]          
            mask_targets[m_i, :, :] = intersect_mask(pr_roi, gt_roi, \
                                                     gt_mask, self.cfg.MAIN_DEFAULT_MASK_SIZE, cfg)
                        
        # keep_inds are use to filter rois as well as masks
        seg_overlaps_all = seg_overlaps_all[keep_inds]
        seg_overlaps_max = seg_overlaps_max[keep_inds]                
        mask_targets = mask_targets[keep_inds]
        mask_target_weights = np.zeros(mask_targets.shape)
        mask_target_weights[0:len(fg_inds_roi), :, :] = 1
        mask_labels = gt_boxes[roi_gt_assignment, 4]
        mask_labels[bg_inds_seg] = 0
        mask_labels = mask_labels[keep_inds]
        mask_labels[len(fg_inds_roi):] = 0
               
        sample_masks = {'seg_overlaps_max': seg_overlaps_max,
                        'seg_overlaps_all': seg_overlaps_all,
                        'seg_gt_assignment': seg_gt_assignment,
                        'sel_masks': (sel_masks >= cfg.MAIN_DEFAULT_BINARIZE_THRESH).astype(bool),
                        'mask_labels': mask_labels, 
                        'mask_targets': (mask_targets >= cfg.MAIN_DEFAULT_BINARIZE_THRESH).astype(bool), 
                        'mask_target_weights': mask_target_weights}
        
        return sample_rois, sample_masks
    
    
    def project_im_rois(self, im_rois, im_scale_factor):
        """Project image RoIs into the rescaled training image."""
        rois = im_rois * im_scale_factor        
        return rois
    
    
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
    cur_idx = property(get_cur_idx, set_cur_idx, del_cur_idx, "cur_idx's docstring")
    perm_ids = property(get_perm_ids, set_perm_ids, del_perm_ids, "perm_ids's docstring")
    images = property(get_images, set_images, del_images, "images's docstring")
    masks = property(get_masks, set_masks, del_masks, "masks's docstring")
    num_cls = property(get_num_cls, set_num_cls, del_num_cls, "num_cls's docstring")
    bbox_means = property(get_bbox_means, set_bbox_means, del_bbox_means, "bbox_means's docstring")
    bbox_stds = property(get_bbox_stds, set_bbox_stds, del_bbox_stds, "bbox_stds's docstring")
    cache_im_dir = property(get_cache_im_dir, set_cache_im_dir, del_cache_im_dir, "cache_im_dir's docstring")
    in_gt_dir = property(get_in_gt_dir, set_in_gt_dir, del_in_gt_dir, "in_gt_dir's docstring")
    in_pr_dir = property(get_in_pr_dir, set_in_pr_dir, del_in_pr_dir, "in_pr_dir's docstring")
    
