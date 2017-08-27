# Project: segm_cfm
# Module: libs.datasets.Pascal
# Written by: Romuald FOTSO
# Based on: py-faster-rcnn 
#    [https://github.com/rbgirshick/py-faster-rcnn]
# Licensed under MIT License
# Date: 17 Jun 2017

import os, pickle, copy
import numpy as np
import os.path as osp
import scipy.io as sio
import xml.etree.ElementTree as ET
from multiprocessing import Process, Queue

from datasets.Dataset import Dataset
from datasets.Image import Image
from datasets.Mask import Mask
from utils.regression import *
from utils.timer import Timer

class Pascal(Dataset):
    
    def __init__(self,
                 name, 
                 path_dir,
                 classes=None,
                 cls_to_id=None,
                 sets=None,
                 year=2007,
                 metric=None, 
                 cfg=None):
        self._year = year
        self._metric = metric
        self._cfg = cfg
        
        super(Pascal, self).__init__(name, 
                 path_dir,
                 classes,
                 cls_to_id,
                 sets,
                 cfg)
        
    def get_cfg(self):
        return self._cfg


    def set_cfg(self, value):
        self._cfg = value


    def del_cfg(self):
        del self._cfg


    def get_year(self):
        return self._year


    def get_metric(self):
        return self._metric


    def set_year(self, value):
        self._year = value


    def set_metric(self, value):
        self._metric = value


    def del_year(self):
        del self._year


    def del_metric(self):
        del self._metric
        
    def built_im_path(self, im_nm, im_DIR):
        im_fn = None
        im_pn = None
        
        for ext in self.cfg.PASCAL_DATASET_DEFAULT_EXT:
            im_pn = osp.join(im_DIR, im_nm+"."+ext)
            if osp.exists(im_pn):
                im_fn = im_nm+"."+ext
                break
            
        assert im_fn != None, \
        "[ERROR] unable to load image {} in {}".format(im_nm, im_DIR)
        
        return im_fn, im_pn 
        
    def load_images(self, im_names):
        images = []
        
        cache_images_file = osp.join(self.cfg.MAIN_DIR_ROOT, "cache","pascal_images.pkl")
        if osp.exists(cache_images_file):
            with open(cache_images_file,'rb') as fp:
                images = pickle.load(fp)
                print ('[INFO] images_obj loaded from {}'.format(cache_images_file))
                fp.close()
            return images
        
        for im_nm in im_names:
            im_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.PASCAL_DATASET_DIR_IMAGE)
            anno_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.PASCAL_DATASET_DIR_ANNOTATION)
           
            im_fn, im_pn = self.built_im_path(im_nm, im_DIR)
            
            anno_pn = osp.join(anno_DIR, im_nm+".xml")
            assert osp.exists(anno_pn), \
                   "[ERROR] unable to load annotation {}".format(im_nm+".xml")
            
            rois = self.readXmlAnno(anno_pn)            
            img = Image(im_fn, im_pn, gt_rois=rois)           
            images.append(img)
            
        # prepare images for training
        images = self.prepare_images(images)
            
        if not osp.exists(cache_images_file):
            with open(cache_images_file,'wb') as fp:
                pickle.dump(images, fp)
                print ('[INFO] images_obj saved to {}'.format(cache_images_file))
                fp.close()
            
        return images
    
        
    def load_gt_masks_rois(self, gt_set, in_gt_dir, num_proc=1):
        masks = []
        images = []        
        im_names = self.sets[gt_set]['im_names']
        
        imgs_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                                  self.cfg.PASCAL_DATASET_DIR_IMAGE)
        
        cache_masks_fn = '{}_gt_{}_masks.pkl'.format(self.name, gt_set)
        cache_images_fn = '{}_gt_{}_images.pkl'.format(self.name, gt_set)        
        
        cache_masks_pn = osp.join(self.cfg.MAIN_DIR_ROOT,
                                       "cache", cache_masks_fn)
        cache_images_pn = osp.join(self.cfg.MAIN_DIR_ROOT,
                                       "cache",cache_images_fn)
        
        print ('[INFO] loading gt masks & rois for {}...'.format(self.name))        
        if osp.exists(cache_masks_pn) and osp.exists(cache_images_pn):
            with open(cache_masks_pn,'rb') as fp:
                masks = pickle.load(fp)
                print ('[INFO] masks with gt loaded from {}'.format(cache_masks_pn))
                fp.close()
                
            with open(cache_images_pn,'rb') as fp:
                images = pickle.load(fp)
                print ('[INFO] images with gt loaded from {}'.format(cache_images_pn))
                fp.close()
                
            return masks, images
        
        # sub-method for a multiprocessing
        def _load_gt_masks_rois(proc_id, l_start, l_end, im_names, queue, cfg):
            _images = []
            _masks = []
            timer = Timer()
            
            timer.tic()            
            for im_i in xrange(l_start, l_end):
                im_nm = im_names[im_i]
                mask_mat_fn = im_nm+'.mat'
                mask_mat_pn = osp.join(in_gt_dir, mask_mat_fn)                              
                
                """ load masks data """                
                assert osp.exists(mask_mat_pn), \
                '[ERROR] unable to find gt MAT file: {},'+ \
                ' did you tools/parse_segm_data.py on your dataset ?'.format(mask_mat_pn)
                
                mask_mat = sio.loadmat(mask_mat_pn)               
                labels = mask_mat['labels'][0]
                bboxes = mask_mat['bboxes']
                superpixels = mask_mat['superpixels']
                                
                mask = Mask(filename=mask_mat_fn,pathname=mask_mat_pn)
                
                im_info = {"width": superpixels.shape[1],
                           "height": superpixels.shape[0],
                           "depth": 1}
                            
                data = {'labels': labels,
                        'bboxes': bboxes,
                        'im_info': im_info,
                        'flipped': False
                        }
                mask.gt_data = data
                _masks.append(mask)
                
                """ load rois data """
                img_fn, img_pn = self.built_im_path(im_nm, imgs_DIR)
                image = Image(filename=img_fn,pathname=img_pn)
                num_masks = len(bboxes)
                boxes = np.zeros((num_masks, 4), dtype=np.uint16)
                roi_gt_classes = np.zeros((num_masks), dtype=np.int32)
                roi_overlaps = np.zeros((num_masks, len(self.cls_to_id)), dtype=np.float32)
                roi_seg_areas = np.zeros((num_masks), dtype=np.float32)           
                
                for m_i, cls_id in enumerate(labels):
                    boxes[m_i] = bboxes[m_i]
                    roi_gt_classes[m_i] = cls_id
                    roi_overlaps[m_i, cls_id] = 1.0
                    roi_seg_areas[m_i] = (bboxes[m_i][2] - bboxes[m_i][0] + 1) \
                                            * (bboxes[m_i][1] - bboxes[m_i][3] + 1)
                
                rois = {
                    'boxes': boxes,
                    'gt_classes': roi_gt_classes,
                    #'overlaps': roi_overlaps,
                    #'seg_areas': roi_seg_areas,
                    'im_info': im_info,
                    'flipped': False
                    }
                image.gt_rois = rois
                _images.append(image)
            
            timer.toc()    
            #n_imgs = len(xrange(l_start, l_end))
            print '[INFO] >> PROC.ID [{}]:  {}-{}/{} images processed in {:.3f}'.\
            format(proc_id, l_start, l_end, len(im_names), timer.average_time)
                
            #return on queue
            queue.put([_masks, _images])           
        
        processes = []
        queues  = []
        num_imgs = len(im_names)
        l_start = 0   
        if num_imgs <= num_proc:
            num_proc = num_imgs
        
        l_offset = int(np.ceil(num_imgs / float(num_proc)))
        
        for proc_id in xrange(num_proc):
            l_end = min(l_start + l_offset, num_imgs)
            q = Queue()
            p = Process(target=_load_gt_masks_rois, 
                        args=(proc_id, l_start, l_end, im_names, q, self.cfg))        
            p.start()
            processes.append(p)
            queues.append(q)
            l_start += l_offset
            
        for proc_id in xrange(num_proc):            
            _masks, _images = queues[proc_id].get()
            masks.extend(_masks)
            images.extend(_images)
            processes[proc_id].join()
            
        print ('gt > bef. flipped, len(masks): {}'.format(len(masks)))
        print ('gt > bef. flipped, len(images): {}'.format(len(images)))
        
        """ append horizontal flipped  images & masks"""
        print ('[INFO] append horizontal flipped  gt images & masks: {}'.format(self.name))
        
        if self.cfg.TRAIN_DEFAULT_USE_FLIPPED and gt_set != 'test':
            images = self.append_flipped_images(images=images, src='gt', num_proc=num_proc)
            masks = self.append_flipped_masks(masks=masks, gt_mat_dir=in_gt_dir, 
                                                src='gt', num_proc=num_proc)
            
        print ('gt > aft. flipped, len(masks): {}'.format(len(masks)))
        print ('gt > aft. flipped, len(images): {}'.format(len(images)))
        
        if not osp.exists(cache_masks_pn):
            with open(cache_masks_pn,'wb') as fp:
                pickle.dump(masks, fp)
                print ('[INFO] masks with gt saved to {}'.format(cache_masks_pn))
                fp.close()
                
        if not osp.exists(cache_images_pn):
            with open(cache_images_pn,'wb') as fp:
                pickle.dump(images, fp)
                print ('[INFO] images with gt saved to {}'.format(cache_images_pn))
                fp.close()
        
        return masks, images
    
    def load_mask_rois_proposals(self, gt_set, in_pr_dir, src, top_k=-1, num_proc=1):
        masks = []
        images = []
        im_names = self.sets[gt_set]['im_names']
        
        imgs_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                                  self.cfg.PASCAL_DATASET_DIR_IMAGE)
            
        cache_masks_fn = '{}_{}_{}_masks.pkl'.format(self.name, src, gt_set)            
        cache_masks_pn = osp.join(self.cfg.MAIN_DIR_ROOT,"cache",cache_masks_fn)
        
        cache_imgs_fn = '{}_{}_{}_images.pkl'.format(self.name, src, gt_set)
        cache_imgs_pn = osp.join(self.cfg.MAIN_DIR_ROOT,"cache",cache_imgs_fn)                                                                   
        
        print ('[INFO] loading proposals masks & rois for {}...'.format(self.name))        
        if osp.exists(cache_masks_pn) and osp.exists(cache_imgs_pn):
            # load masks
            with open(cache_masks_pn,'rb') as fp:
                masks = pickle.load(fp)
                print ('[INFO] masks with pr loaded from {}'.format(cache_masks_pn))
                fp.close()
            
            # load images
            with open(cache_imgs_pn,'rb') as fp:
                images = pickle.load(fp)
                print ('[INFO] images with pr loaded from {}'.format(cache_imgs_pn))
                fp.close()
                
            return masks, images
        
        # sub-method for a multiprocessing
        def _load_mask_rois_proposals(proc_id, l_start, l_end, im_names, top_k, queue, cfg):
            _images = []
            _masks = []
            timer = Timer()
            
            timer.tic()            
            for im_i in xrange(l_start, l_end):
                im_nm = im_names[im_i]
                pr_mat_fn = im_nm+'.mat'
                pr_mat_pn = osp.join(in_pr_dir, pr_mat_fn)
                                            
                """ load masks data """                
                assert osp.exists(pr_mat_pn), \
                '[ERROR] unable to find proposal MAT file: {}, d'+ \
                'did you call tools/generate_data.py on your dataset ?'.format(pr_mat_pn)
                                 
                pr_mat = sio.loadmat(pr_mat_pn)
                if src == 'selective_search':     
                    labels = pr_mat['labels'][0]
                    bboxes = pr_mat['bboxes']
                    #superpixels = pr_mat['superpixels']
                    im_shape = pr_mat['superpixels'].shape
                    
                elif src == 'mcg':                   
                    labels = pr_mat['labels'][0]
                    bboxes = pr_mat['bboxes']
                    #superpixels = pr_mat['superpixels']
                    im_shape = pr_mat['superpixels'].shape[1:3]
                    
                elif src == 'cob':
                    labels = pr_mat['labels'][0]
                    bboxes = pr_mat['bboxes']
                    #superpixels = pr_mat['superpixels']
                    im_shape = pr_mat['superpixels'].shape[1:3]
                    
                else:
                    raise NotImplemented
                            
                mask = Mask(filename=pr_mat_fn,pathname=pr_mat_pn)
                
                im_info = {"width": im_shape[1],
                           "height": im_shape[0],
                           "depth": 1}
                               
                data = {'labels': labels,
                        'bboxes': bboxes,
                        'im_info': im_info,
                        'flipped': False
                    }
                mask.pr_data = data
                _masks.append(mask)     
                
                """ load rois data """
                img_fn, img_pn = self.built_im_path(im_nm, imgs_DIR)
                image = Image(filename=img_fn,pathname=img_pn)
                
                rois = {
                    'boxes': bboxes,
                    'im_info': im_info,
                    'flipped': False
                    }
                image.pr_rois = rois
                _images.append(image)
            
            timer.toc()    
            #n_imgs = len(xrange(l_start, l_end))
            print '[INFO] >> PROC.ID [{}]:  {}-{}/{} images processed in {:.3f}'.\
            format(proc_id, l_start, l_end, len(im_names), timer.average_time) 
               
            #return on queue
            queue.put([_masks, _images])            
        
        processes = []
        queues  = []
        num_imgs = len(im_names)
        l_start = 0
        if num_imgs <= num_proc:
            num_proc = num_imgs            
        l_offset = int(np.ceil(num_imgs / float(num_proc)))
        
        for proc_id in xrange(num_proc):
            l_end = min(l_start + l_offset, num_imgs)
            q = Queue()
            p = Process(target=_load_mask_rois_proposals, 
                        args=(proc_id, l_start, l_end, im_names, top_k, q, self.cfg))        
            p.start()
            processes.append(p)
            queues.append(q)
            l_start += l_offset
            
        for proc_id in xrange(num_proc):            
            _masks, _images = queues[proc_id].get()
            masks.extend(_masks)
            images.extend(_images)
            processes[proc_id].join()
        
        """ append horizontal flipped  images & masks"""
        print ('[INFO] append horizontal flipped  pr images & masks: {}'.format(self.name)) 
        
        if self.cfg.TRAIN_DEFAULT_USE_FLIPPED and gt_set != 'test':
            images = self.append_flipped_images(images=images, src='pr', num_proc=num_proc)
            masks = self.append_flipped_masks(masks=masks, pr_mat_dir=in_pr_dir, 
                                                    src='pr', num_proc=num_proc)
        
        if not osp.exists(cache_masks_pn):
            with open(cache_masks_pn,'wb') as fp:
                pickle.dump(masks, fp)
                print ('[INFO] masks with pr saved to {}'.format(cache_masks_pn))
                fp.close()
                
        if not osp.exists(cache_imgs_pn):
            with open(cache_imgs_pn,'wb') as fp:
                pickle.dump(images, fp)
                print ('[INFO] images with pr saved to {}'.format(cache_imgs_pn))
                fp.close()               
        
        return masks, images
    
    
    def merge_gt_proposal_rois(self, gt_images, pr_images):
        all_images = []            
        for gt_image, pr_image in zip(gt_images, pr_images):
            image = copy.deepcopy(gt_image)
            image.pr_rois = pr_image.pr_rois
            all_images.append(image)
        del gt_images
        del pr_images
        return all_images       

    
    def merge_gt_proposal_masks(self, gt_masks, pr_masks):
        all_masks = []            
        for gt_mask, pr_mask in zip(gt_masks, pr_masks):
            mask = copy.deepcopy(gt_mask)
            mask.pr_data = pr_mask.pr_data
            all_masks.append(mask)
        del gt_masks
        del pr_masks
        return all_masks


    def load_sets(self):
        sets = {
            "train":    {"im_names": [], "images": [], "num_items":0},
            "trainval": {"im_names": [], "images": [], "num_items":0},
            "val":      {"im_names": [], "images": [], "num_items":0},
            "test":     {"im_names": [], "images": [], "num_items":0}
        }
        task = self.cfg.MAIN_DEFAULT_TASK
                
        if task == 'CFC' or task == 'DET':
            DATASET_DIR = self.cfg.PASCAL_DATASET_DIR_MAIN_SET
        elif task == 'SEGM':
            DATASET_DIR = self.cfg.PASCAL_DATASET_DIR_SEGM_SET
        else:
            raise('[ERROR] unknown task')
        
        if self.name == "bsd_voc2012":
            DATASET_DIR = ''
        
        train_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_TRAIN)
        
        trainval_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_TRAINVAL)
        
        test_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_TEST)
        
        val_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_VAL)        
        
        im_names = []
        if osp.exists(train_file):
            with open(train_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["train"]["im_names"] = im_names
            sets["train"]["num_items"] = len(im_names)            
        else:
            print ("[WARN] unable to load file {}".format(train_file))
        del im_names
        
        im_names = []
        if osp.exists(trainval_file):
            with open(trainval_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["trainval"]["im_names"] = im_names
            sets["trainval"]["num_items"] = len(im_names)            
        else:
            print ("[WARN] unable to load file {}".format(trainval_file))
        del im_names
        
        im_names = []
        if osp.exists(test_file):
            with open(test_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["test"]["im_names"] = im_names
            sets["test"]["num_items"] = len(im_names)
        else:
            print ("[WARN] unable to load file {}".format(test_file))
        del im_names
        
        im_names = []
        if osp.exists(val_file):
            with open(val_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["val"]["im_names"] = im_names
            sets["val"]["num_items"] = len(im_names)
        else:
            print ("[WARN] unable to load file {}".format(val_file))
        del im_names
        
        self.sets = sets
        
    def readXmlAnno(self, anno_pn):
        tree = ET.parse(anno_pn)
        root = tree.getroot()
            
        objs = root.findall('object')
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, len(self.cls_to_id)), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        is_segm = int(root.find('segmented').text)
        size = root.find('size')
        im_info = {"width": int(size.find('width').text),
                   "height": int(size.find('height').text),
                   "depth": int(size.find('depth').text)}
                    
        for id_obj, obj in enumerate(objs):
            xmin = int(float(obj.find('bndbox').find('xmin').text)) - 1
            ymin = int(float(obj.find('bndbox').find('ymin').text)) - 1
            xmax = int(float(obj.find('bndbox').find('xmax').text)) - 1
            ymax = int(float(obj.find('bndbox').find('ymax').text)) - 1
            id_cls = self.cls_to_id[obj.find('name').text.strip()]
            
            boxes[id_obj,:] = [xmin, ymin, xmax, ymax]
            gt_classes[id_obj] = id_cls
            overlaps[id_obj, id_cls] = 1.0
            seg_areas[id_obj] = (xmax - xmin + 1) * (ymax - ymin + 1)
        
        #overlaps = csr_matrix(overlaps)
        
        return {"boxes": boxes,
                "gt_classes": gt_classes,
                "gt_overlaps": overlaps,
                "seg_areas": seg_areas,
                "is_segm": is_segm,
                "im_info": im_info,
                "flipped": False}
        
    def filter_images(self, images):
        num_imgs = len(images)
        
        images_OK = []
        
        for im_i in xrange(num_imgs):
            gt_overlaps = images[im_i].gt_rois['max_overlaps']
            pr_overlaps = images[im_i].pr_rois['max_overlaps']            
            all_overlaps = gt_overlaps.extend(pr_overlaps)
            
            # find boxes with sufficient overlap
            fg_inds = np.where(all_overlaps >= self.cfg.TRAIN_DEFAULT_FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((all_overlaps < self.cfg.TRAIN_DEFAULT_BG_THRESH_HI) &
                               (all_overlaps >= self.cfg.TRAIN_DEFAULT_BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            
            if valid:
                images_OK.append(images[im_i])
        
        num_after = len(images_OK)       
        print '[INFO] Filtered {} images: {} -> {}'.format(num_imgs - num_after,
                                                    num_imgs, num_after)
        return images_OK
            
        
    def append_flipped_images(self, images, src, num_proc=1):
        num_imgs = len(images)
        widths = []
        
        assert src in ['gt','pr'], \
        '[ERROR] unknown rois src provided'
        
        if src == 'gt':
            widths = [image.gt_rois['im_info']['width'] for image in images]
        elif src == 'pr':
            widths = [image.pr_rois['im_info']['width'] for image in images]
        
        # sub-method for multiprocessing
        def _append_flipped_images(proc_id, l_start, l_end, queue, cfg):
            _images_flip = []
            timer = Timer()
            
            timer.tic() 
            for im_i in xrange(l_start, l_end):
                if src == 'gt':
                    boxes = images[im_i].gt_rois['boxes'].copy()
                elif src == 'pr':
                    boxes = images[im_i].pr_rois['boxes'].copy()
                    
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = widths[im_i] - oldx2 - 1
                boxes[:, 2] = widths[im_i] - oldx1 - 1            
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                assert (boxes[:, 0] >= 0).all()
                assert (boxes[:, 2] >= 0).all()
                            
                image = copy.deepcopy(images[im_i])
                if src == 'gt':
                    image.gt_rois['boxes'] = boxes
                    image.gt_rois['flipped'] = True
                elif src == 'pr':
                    image.pr_rois['boxes'] = boxes
                    image.pr_rois['flipped'] = True
                    
                _images_flip.append(image)
                
            timer.toc()    
            #n_imgs = len(xrange(l_start, l_end))
            print '[INFO] >> PROC.ID [{}]:  {}-{}/{} images flipped in {:.3f}'.\
            format(proc_id, l_start, l_end, len(images), timer.average_time)
            
            #return on queue
            queue.put(_images_flip)
            
        processes = []
        queues = []
        l_start = 0
        if num_imgs <= num_proc:
            num_proc = num_imgs
            
        l_offset = int(np.ceil(num_imgs / float(num_proc)))
        
        for proc_id in xrange(num_proc):
            l_end = min(l_start + l_offset, num_imgs)
            q = Queue()
            p = Process(target=_append_flipped_images, 
                        args=(proc_id, l_start, l_end, q, self.cfg))        
            p.start()
            processes.append(p)
            queues.append(q)
            l_start += l_offset
        
        for proc_id in xrange(num_proc):
            images.extend(queues[proc_id].get())
            processes[proc_id].join()
                        
        return images

    
    def append_flipped_masks(self, masks, gt_mat_dir=None, pr_mat_dir=None, src=None, num_proc=1):
        num_masks = len(masks)
        
        assert src in ['gt','pr'], \
        '[ERROR] unknown masks src provided'
                        
        def _append_flipped_masks(proc_id, l_start, l_end, queue, cfg):
            _masks_flip = []
            timer = Timer()
            
            timer.tic()
            for m_i in xrange(l_start, l_end):
                if gt_mat_dir != None:
                    gt_mat_flipped_fn = (masks[m_i].filename).split('.')[0]+'_flipped.mat'
                    gt_mat_flipped_pn = osp.join(gt_mat_dir, gt_mat_flipped_fn)                
                if pr_mat_dir != None:
                    pr_mat_flipped_fn = (masks[m_i].filename).split('.')[0]+'_flipped.mat'
                    pr_mat_flipped_pn = osp.join(pr_mat_dir, pr_mat_flipped_fn)
                                
                mask = copy.deepcopy(masks[m_i])
                
                if src == 'gt' and osp.exists(gt_mat_flipped_pn): 
                    gt_mat_flipped = sio.loadmat(gt_mat_flipped_pn)                    
                    mask.gt_data['bboxes'] = gt_mat_flipped['bboxes']
                    mask.gt_data['flipped'] = True
                    del gt_mat_flipped                  
                    _masks_flip.append(mask)
                    
                elif src == 'pr' and osp.exists(pr_mat_flipped_pn):
                    pr_mat_flipped = sio.loadmat(pr_mat_flipped_pn)
                    mask.pr_data['bboxes'] = pr_mat_flipped['bboxes']
                    mask.pr_data['flipped'] = True
                    del pr_mat_flipped                       
                    _masks_flip.append(mask)
                    
                else:                                        
                    if src == 'gt':
                        gt_mat_fn = (masks[m_i].filename).split('.')[0]+'.mat'
                        gt_mat_pn = osp.join(gt_mat_dir, gt_mat_fn)
                        assert osp.exists(gt_mat_pn), \
                        '[ERROR] missing mat file: {}'.format(gt_mat_pn) 
                        
                        gt_mat = sio.loadmat(gt_mat_pn)                    
                        superpixels = gt_mat['superpixels']  
                        gt_inst_ids = gt_mat['inst_ids']               
                        del gt_mat
                    elif src == 'pr':
                        pr_mat_fn = (masks[m_i].filename).split('.')[0]+'.mat'
                        pr_mat_pn = osp.join(pr_mat_dir, pr_mat_fn)
                        assert osp.exists(pr_mat_pn), \
                        '[ERROR] missing mat file: {}'.format(pr_mat_pn) 
                        
                        pr_mat = sio.loadmat(pr_mat_pn)
                        superpixels = pr_mat['superpixels']
                        del pr_mat
                        
                    # flip the mask instances
                    if len(superpixels.shape) == 2:
                        superpixels = superpixels[:, ::-1]
                    elif len(superpixels.shape) == 3:
                        superpixels = superpixels[:, :, ::-1]
                    else:
                        raise Exception('[ERROR] incorrect shape for superpixels: {}'.format(superpixels.shape))
                    
                    if src == 'gt':                  
                        boxes = mask.gt_data['bboxes']
                        width = mask.gt_data['im_info']['width']
                        #print 'bef. boxes: {}'.format(boxes)
                        
                        # flip the bboxes
                        oldx1 = boxes[:, 0].copy()
                        oldx2 = boxes[:, 2].copy()
                        boxes[:, 0] = width - oldx2 - 1
                        boxes[:, 2] = width - oldx1 - 1            
                        assert (boxes[:, 2] >= boxes[:, 0]).all()
                        assert (boxes[:, 0] >= 0).all()
                        assert (boxes[:, 2] >= 0).all()
                        
                        mask.gt_data['bboxes'] = boxes
                        mask.gt_data['flipped'] = True
                        #print 'aft. boxes: {}'.format(boxes)
                             
                        gt_mat_flipped = {
                            "labels": mask.gt_data['labels'],
                            "bboxes": boxes,
                            "inst_ids": gt_inst_ids,
                            "superpixels": superpixels}                   
                        
                        try:
                            sio.savemat(gt_mat_flipped_pn, gt_mat_flipped, do_compression=True)
                        except Exception as e:
                            print ("[WARM] exception found: {}, trying sio.savemat without compression".format(e))
                            sio.savemat(gt_mat_flipped_pn, gt_mat_flipped, do_compression=False)
                        except Exception as f:            
                            print ("[ERROR] exception found: {}".format(f))
                        
                    elif src == 'pr':
                        boxes = mask.pr_data['bboxes']
                        width = mask.pr_data['im_info']['width']
                        
                        # flip the bboxes
                        oldx1 = boxes[:, 0].copy()
                        oldx2 = boxes[:, 2].copy()
                        boxes[:, 0] = width - oldx2 - 1
                        boxes[:, 2] = width - oldx1 - 1            
                        assert (boxes[:, 2] >= boxes[:, 0]).all()
                        assert (boxes[:, 0] >= 0).all()
                        assert (boxes[:, 2] >= 0).all()
                        
                        mask.pr_data['bboxes'] = boxes
                        mask.pr_data['flipped'] = True
                                   
                        pr_mat_flipped = {
                            "labels": mask.pr_data['labels'],
                            "bboxes": boxes,                        
                            "superpixels": superpixels}
                            
                        try:
                            sio.savemat(pr_mat_flipped_pn, pr_mat_flipped, do_compression=True)
                        except Exception as e:
                            print ("[WARM] exception found: {}, trying sio.savemat without compression".format(e))
                            sio.savemat(pr_mat_flipped_pn, pr_mat_flipped, do_compression=False)
                        except Exception as f:            
                            print ("[ERROR] exception found: {}".format(f))                   
                        
                    _masks_flip.append(mask)              
            
            timer.toc()    
            #n_masks = len(xrange(l_start, l_end))
            print '[INFO] >> PROC.ID [{}]:  {}-{}/{} masks flipped in {:.3f}'.\
            format(proc_id, l_start, l_end, len(masks), timer.average_time)
            
            #return on queue
            queue.put(_masks_flip)
                
        processes = []
        queues = []
        l_start = 0   
        if num_masks <= num_proc:
            num_proc = num_masks
        l_offset = int(np.ceil(num_masks / float(num_proc)))
        
        for proc_id in xrange(num_proc):
            l_end = min(l_start + l_offset, num_masks)
            q = Queue()
            p = Process(target=_append_flipped_masks, 
                        args=(proc_id, l_start, l_end, q, self.cfg))        
            p.start()
            processes.append(p)
            queues.append(q)
            l_start += l_offset
                    
        for proc_id in xrange(num_proc):
            _masks_flip = queues[proc_id].get()
            masks.extend(_masks_flip)
            processes[proc_id].join()
                                
        return masks
    
    
    def built_output_dir(self, root_dir, phase):
        output_dir = osp.join(root_dir, self.cfg.MAIN_DIR_OUTPUTS, self.name, 
                              phase+'_'+self.cfg.TRAIN_DEFAULT_SEGM_METHOD)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        return output_dir

    year = property(get_year, set_year, del_year, "year's docstring")
    metric = property(get_metric, set_metric, del_metric, "metric's docstring")
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
    
    
        
    
