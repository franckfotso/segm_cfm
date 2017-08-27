# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: segm_cfm
# Module: tools.parse_segm_data
# Copyright (c) 2017
# Written by: Romuald FOTSO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: python tools/parse_gt_data.py --dataset voc_2007 \
#        --gt_set train --in_seg_cls data/voc_2007/SegmentationClass \
#        --in_seg_inst data/voc_2007/SegmentationObject --num_proc 1 \
#        --out_seg_dir data/sbd_voc2007/SEGM-RAW
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import _init_paths
import argparse, pprint, os, cv2
import scipy.io as sio
import os.path as osp
import numpy as np

from Config import Config
from datasets.Pascal import Pascal
from multiprocessing import Process
from utils.timer import Timer

def parse_args():
    """
        construct the argument parse and parse the arguments
    """    
    ap = argparse.ArgumentParser(description='Prepare data for training')
    ap.add_argument('--dataset', dest='dataset', required=True,
                        help='dataset name to use',
                        default='voc_2007', type=str)
    ap.add_argument('--gt_set', dest='gt_set', required=True,
                        help='gt set use to list data',
                        default='train', type=str)
    ap.add_argument('--in_seg_cls', dest='in_seg_cls', 
                        help='input directory of segments per class',
                        default='data/voc_2007/SegmentationClass', type=str)
    ap.add_argument('--in_seg_inst', dest='in_seg_inst',
                        help='input directory of segments per instance',
                        default='data/voc_2007/SegmentationObject', type=str)
    ap.add_argument('--num_proc', dest='num_proc', 
                        help='numbers of concurrent processes',
                        default=1, type=int)
    ap.add_argument('--out_seg_dir', dest='out_seg_dir', required=True,
                        help='output directory of segments',
                        default='data/sbd_voc2007/SEGM-RAW', type=str)
    return vars(ap.parse_args())


def parse_from_pascal(l_start, l_end, dataset, gt_set,
                          in_seg_dirs, out_seg_dir):
    in_seg_cls, in_seg_obj = in_seg_dirs
           
    im_names = dataset.sets[gt_set]["im_names"]
    timer = Timer()
        
    for im_i in xrange(l_start, l_end):
        im_nm =  im_names[im_i]
        _, mask_cls_pn = dataset.built_im_path(im_nm, in_seg_cls)
        _, mask_obj_pn = dataset.built_im_path(im_nm, in_seg_obj)
        out_mat_pn = osp.join(out_seg_dir, im_nm+'.mat')
        timer.tic()
        
        if osp.exists(out_mat_pn):
            continue
            
        mask_cls_RAW = cv2.imread(mask_cls_pn)
        mask_obj_RAW = cv2.imread(mask_obj_pn)
        
        mask_cls_RAW = cv2.cvtColor(mask_cls_RAW, cv2.COLOR_BGR2RGB)
        mask_obj_RAW = cv2.cvtColor(mask_obj_RAW, cv2.COLOR_BGR2RGB)
        cls_R,cls_G,cls_B = cv2.split(mask_cls_RAW)
        obj_R,obj_G,obj_B = cv2.split(mask_obj_RAW)           
        
        obj_R_uniq_vals = np.unique(obj_R)
        obj_G_uniq_vals = np.unique(obj_G)
        obj_B_uniq_vals = np.unique(obj_B)
                
        """ load masks data """         
        superpixels = np.zeros((mask_cls_RAW.shape[0],mask_cls_RAW.shape[1]),dtype=np.float)
        labels = []
        bboxes = []
        inst_ids = []
        num_uniq_vals = len(obj_R_uniq_vals)*len(obj_G_uniq_vals)*len(obj_B_uniq_vals)
        num_ids = max(min(num_uniq_vals,255),255)
        pool_ids = np.random.permutation(xrange(1,num_ids))         
        idx = 0
        
        for obj_R_val in obj_R_uniq_vals:
            for obj_G_val in obj_G_uniq_vals:
                for obj_B_val in obj_B_uniq_vals:
                    [row_ids, col_ids] = np.array(np.where((obj_R == obj_R_val) & \
                                                (obj_G == obj_G_val) & \
                                                (obj_B == obj_B_val)))
                    
                    if len(row_ids) > 0 and len(row_ids)==len(col_ids):
                        rgb_y, rgb_x = (row_ids[0],col_ids[0])
                        cls_R_val = str(cls_R[rgb_y, rgb_x])
                        cls_G_val = str(cls_G[rgb_y, rgb_x])
                        cls_B_val = str(cls_B[rgb_y, rgb_x])
                        
                        rgb_str = "_".join([cls_R_val, cls_G_val, cls_B_val])
                        inst_cls = dataset.rgb_to_cls[rgb_str]
                        
                        # save only mask_inst with valid class
                        if inst_cls in dataset.cls_to_id.keys():
                            label = dataset.cls_to_id[inst_cls]
                            inst_id = pool_ids[idx]
                            superpixels[row_ids,col_ids] = inst_id
                            
                            y1 = int(np.min(row_ids))
                            x1 = int(np.min(col_ids))
                            y2 = int(np.max(row_ids))
                            x2 = int(np.max(col_ids)) 
                            box = np.array([x1,y1,x2,y2])
                                                     
                            labels.append(label)
                            inst_ids.append(inst_id)
                            bboxes.append(box)
                            idx += 1
        
        mask_mat = {"labels": np.array(labels),
                    "bboxes": np.array(bboxes),
                    "inst_ids": np.array(inst_ids),
                    "superpixels": np.array(superpixels)}
        
        # save mat-mask output        
        sio.savemat(out_mat_pn, mask_mat, do_compression=True)
        timer.toc()
        print '[INFO] %d/%d use time (s) %f' % (im_i, len(im_names), timer.average_time)
        
        
def parse_from_coco(l_start, l_end, dataset, gt_set,
                          in_seg_dirs, out_seg_dir):
    raise NotImplemented


def parse_from_bsd(l_start, l_end, dataset, gt_set,
                          in_seg_dirs, out_seg_dir):
    in_seg_cls, in_seg_inst = in_seg_dirs
        
    im_names = dataset.sets[gt_set]["im_names"]
    timer = Timer()
    
    for im_i in xrange(l_start, l_end):
        im_nm =  im_names[im_i]
        mask_cls_pn = osp.join(in_seg_cls, im_nm+'.mat')
        mask_inst_pn = osp.join(in_seg_inst, im_nm+'.mat')
        out_mat_pn = osp.join(out_seg_dir, im_nm+'.mat')
        timer.tic()
        
        if osp.exists(out_mat_pn):
            continue
            
        mask_cls_mat = sio.loadmat(mask_cls_pn)
        mask_inst_mat = sio.loadmat(mask_inst_pn)
        
        inst_seg = mask_inst_mat['GTinst']['Segmentation'][0,0]
        cls_seg = mask_cls_mat['GTcls']['Segmentation'][0,0]
        inst_uniq_vals = np.unique(inst_seg)      
        
        labels = []
        bboxes = []
        inst_ids = []
        num_ids = max(min(len(inst_uniq_vals),255),255)
        pool_ids = np.random.permutation(xrange(1,num_ids))
        superpixels = np.zeros((inst_seg.shape[0],inst_seg.shape[1]),dtype=np.float)      
        idx = 0
        
        for inst_uniq_val in inst_uniq_vals:
            [row_ids, col_ids] = np.array(np.where((inst_seg == inst_uniq_val)))
                                          
            if len(row_ids) > 0 and len(row_ids)==len(col_ids):
                rgb_y, rgb_x = (row_ids[0],col_ids[0])
                cls_val = cls_seg[rgb_y, rgb_x]
                inst_id = pool_ids[idx]
                superpixels[row_ids,col_ids] = inst_id
                
                y1 = int(np.min(row_ids))
                x1 = int(np.min(col_ids))
                y2 = int(np.max(row_ids))
                x2 = int(np.max(col_ids)) 
                box = np.array([x1,y1,x2,y2])
                                         
                labels.append(cls_val)
                inst_ids.append(inst_id)
                bboxes.append(box)
                idx += 1
                
        mask_mat = {"labels": np.array(labels),
                    "bboxes": np.array(bboxes),
                    "inst_ids": np.array(inst_ids),
                    "superpixels": np.array(superpixels)}
        
        # save mat-mask output        
        sio.savemat(out_mat_pn, mask_mat, do_compression=True)
        timer.toc()
        print '[INFO] %d/%d use time (s) %f' % (im_i, len(im_names), timer.average_time)
        
        
if __name__ == '__main__':
    args = parse_args()
    
    print('Input args:')
    pprint.pprint(args)
    
    # setup & load configs
    _C = Config(config_pn="config/config.ini")
    cfg = _C.cfg
    
    print('Using config:')
    pprint.pprint(cfg)
    
    print ('[INFO] loading {} dataset for training...'.format(args["dataset"]))
    dataset_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])
    
    dataset = None
    ds_pascal = ["demo", "voc_2007", "voc_2012", "bsd_voc2012"]
    ds_coco = []
    
    if args["dataset"] in ds_pascal:
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
    assert dataset != None, \
        "[ERROR] unable to build {} dataset. Available: {}".\
        format(args["dataset"], ds_pascal)
    
    dataset.load_sets()
    print ('[INFO] dataset.name: {}'.format(dataset.name))
    print ('[INFO] dataset.num_cls: {}'.format(dataset.num_cls))
    print ('[INFO] dataset.train: {}'.format(dataset.sets["train"]["num_items"]))
    print ('[INFO] dataset.trainval: {}'.format(dataset.sets["trainval"]["num_items"]))
    print ('[INFO] dataset.test: {}'.format(dataset.sets["test"]["num_items"]))
    print ('[INFO] dataset.val: {}'.format(dataset.sets["val"]["num_items"]))
    
    in_seg_cls = args['in_seg_cls']
    in_seg_inst = args['in_seg_inst']    
    out_seg_dir = args['out_seg_dir']
    gt_set = args['gt_set']
    max_pr = args['top_k']
    
    # prepare process with multi-process
    processes = []
    num_proc = args['num_proc']
    l_start = 0
    
    assert gt_set in dataset.sets.keys(), \
    '[ERROR] unknown gt_set defined {}'.format(gt_set)
    
    im_names = dataset.sets[gt_set]["im_names"]
    num_imgs = dataset.sets[gt_set]["num_items"]
    l_offset = int(np.ceil(num_imgs / float(num_proc)))
    
    if not osp.exists(out_seg_dir):
        os.mkdir(out_seg_dir)
    
    for proc_id in xrange(num_proc):
        l_end = min(l_start + l_offset, num_imgs)
        
        if args["dataset"] in ['voc_2007', 'voc_2012', 'voc_2007_2012']:
            in_seg_dirs = [in_seg_cls,in_seg_inst]
            p = Process(target=parse_from_pascal, 
                        args=(l_start, l_end, dataset, gt_set,
                              in_seg_dirs, out_seg_dir))
        
        elif args["dataset"] in ['bsd_voc2007', 'bsd_voc2012']:
            in_seg_dirs = [in_seg_cls,in_seg_inst]
            p = Process(target=parse_from_bsd, 
                        args=(l_start, l_end, dataset, gt_set,
                              in_seg_dirs, out_seg_dir))
                
        elif args["dataset"] in ['coco_2014', 'coco_2015']:
            raise NotImplementedError
        else:
            raise Exception('[ERROR] unknown dataset: {} '.format(args['dataset']))
        
        p.start()
        processes.append(p)
        l_start += l_offset
        
    for p in processes:
        p.join()
    
    
    
    