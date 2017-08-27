# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: segm_cfm
# Module: tools.generate_data
# Written by: Romuald FOTSO
# Licensed under MIT License
# Date: 19 Jul 2017
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: python tools/generate_data.py --dataset voc_2007 \
#        --gt_set train --pr_meth selective_search --num_proc 1 \
#        --mode fast --output_im data/sbd_voc2007/SS-IMG \
#        --data IMG--output_mat data/sbd_voc2007/SS-RAW
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import _init_paths
import argparse, pprint, os, cv2, random
import os.path as osp
import numpy as np
import scipy.io as sio
import selective_search
import features

from Config import Config
from datasets.Pascal import Pascal
from multiprocessing import Process
from utils.timer import Timer


def parse_args():
    """
        construct the argument parse and parse the arguments
    """    
    ap = argparse.ArgumentParser(description='Generate data with proposals')
    ap.add_argument('--dataset', dest='dataset',
                        help='dataset name to use',
                        default='voc_2007', type=str)
    ap.add_argument('--gt_set', dest='gt_set',
                        help='gt set use to list data',
                        default='train', type=str)
    ap.add_argument('--pr_meth', dest='pr_meth', required=True,
                        help='proposal method to use', type=str)
    ap.add_argument('--top_k', dest='top_k',
                        help='top K proposals to select',
                        default=-1, type=int)
    ap.add_argument('--mode', dest='mode', required=True,
                        help='processing mode',
                        default='fast', type=str)
    ap.add_argument('--data', dest='data', required=True,
                        help='type of data to build',
                        default='MASK', type=str)
    ap.add_argument('--num_proc', dest='num_proc', 
                        help='numbers of concurrent processes',
                        default=1, type=int)
    ap.add_argument('--output_im', dest='im_out_dir', required=True,
                        help='output directory for images results', type=str)
    ap.add_argument('--output_mat', dest='mat_out_dir', required=True,
                        help='output directory for MAT results', type=str)
    return vars(ap.parse_args())

def generate_with_selective_search_3d(l_start, l_end, dataset, gt_set,
                                   im_in_DIR, im_out_DIR, mat_out_DIR, data, cfg):
    """
        generate the proposals with the selective search method
    """  
        
    assert data in ['MASK','MAT'], \
    '[ERROR] unknow type of data: {}'.format(data)       
        
    im_names = dataset.sets[gt_set]["im_names"]
    timer = Timer()
        
    for im_i in xrange(l_start, l_end):
        im_nm =  im_names[im_i]
        im_out_pn = osp.join(im_out_DIR, im_nm+'.png')
        mat_out_pn = osp.join(mat_out_DIR, im_nm+'.mat')
        _, im_pn = dataset.built_im_path(im_nm, im_in_DIR)        
        
        im_RAW = None     
        timer.tic()
        
        if data == 'MASK' and not osp.exists(im_out_pn):        
            im_RAW = cv2.imread(im_pn)
            im_RAW = cv2.cvtColor(im_RAW, cv2.COLOR_BGR2RGB)
            
            mask = features.SimilarityMask('size' in cfg.SEGM_SS_FEATURE, 
                                           'color' in cfg.SEGM_SS_FEATURE, 
                                           'texture' in cfg.SEGM_SS_FEATURE, 
                                           'fill' in cfg.SEGM_SS_FEATURE)
            
            (R, F, _) = selective_search.hierarchical_segmentation(im_RAW, cfg.SEGM_SS_K, mask)                
            colors = generate_color_table(R)               
            is_mask = False
            
            for depth, label in enumerate(F):
                result = colors[label]
                result = (result*cfg.SEGM_SS_ALPHA + 
                      im_RAW*(1.-cfg.SEGM_SS_ALPHA)).astype(np.uint8)
                
                # extract on targeted depth
                if depth == cfg.SEGM_SS_DEPTH:
                    mask_RAW = result
                    is_mask = True
                    break
                
            assert is_mask == True, "[ERROR] unable to extract segm on depth {}".format(im_nm)
            
            # save raw-img output
            mask_RAW = np.array(mask_RAW, dtype=np.int8)
            #print 'mask_RAW.shape: {}'.format(mask_RAW.shape) 
            cv2.imwrite(im_out_pn, mask_RAW)
            timer.toc()
            print '[INFO] %d/%d use time (s) %f' % (im_i+1, len(im_names), timer.average_time)
        
        """ parse proposals as a MAT for easy handle """
        if data == 'MAT' and not osp.exists(mat_out_pn):  
            assert osp.exists(im_out_pn), \
            '[ERROR] missing MASK for {}. Please generate MASK first.'.format(im_out_pn)
            #print 'im_out_pn: {}'.format(im_out_pn)     
            
            mask_RAW = cv2.imread(im_out_pn)
            mask_RAW = cv2.cvtColor(mask_RAW, cv2.COLOR_BGR2RGB)
                
            R,G,B = cv2.split(mask_RAW)        
            R_uniq_vals = np.unique(R)
            G_uniq_vals = np.unique(G)
            B_uniq_vals = np.unique(B)
            #print 'len(R_uniq_vals): {}'.format(len(R_uniq_vals))
            #print 'len(G_uniq_vals): {}'.format(len(G_uniq_vals))
            #print 'len(B_uniq_vals): {}'.format(len(B_uniq_vals))
            
            labels = []
            bboxes = []
            superpixels = np.zeros((mask_RAW.shape[0],mask_RAW.shape[1]),dtype=np.float)        
            #num_ids = min(len(R_uniq_vals)*len(G_uniq_vals)*len(B_uniq_vals),255)
            num_ids = len(R_uniq_vals)*len(G_uniq_vals)*len(B_uniq_vals)
            labels_ids = np.random.permutation(xrange(1,num_ids+1))
            #print 'num_ids: {}'.format(num_ids)
            #print 'len(labels_ids): {}'.format(len(labels_ids))
            #print 'labels_ids: {}'.format(labels_ids)
            idx = 0
            
            for R_val in R_uniq_vals:
                for G_val in G_uniq_vals:
                    for B_val in B_uniq_vals:
                        [row_ids, col_ids] = np.array(np.where((R == R_val) & \
                                                               (G == G_val) & \
                                                               (B == B_val)))
                        
                        if len(row_ids) > 0 and len(row_ids)==len(col_ids):                        
                            y1 = np.min(row_ids)
                            x1 = np.min(col_ids)
                            y2 = np.max(row_ids)
                            x2 = np.max(col_ids)
                            box = np.array([x1, y1, x2, y2])
                            
                            assert idx < num_ids, \
                            '[ERROR] idx out labels range: {} vs. {}'.format(idx,num_ids)
                            
                            label = labels_ids[idx]                                            
                            superpixels[row_ids,col_ids] = label
                                                          
                            labels.append(label)
                            bboxes.append(box)
                            idx += 1
                                       
            im_mat = {"labels": labels,
                      "bboxes": bboxes, 
                      "superpixels": superpixels}
            
            # save mat-img output        
            sio.savemat(mat_out_pn, im_mat, do_compression=True)
            timer.toc()
            print '[INFO] %d/%d use time (s) %f' % (im_i+1, len(im_names), timer.average_time)
                                
def generate_with_selective_search_1d(l_start, l_end, dataset, gt_set,
                                   im_in_DIR, im_out_DIR, mat_out_DIR, data, cfg):
    """
        generate the proposals with the selective search method
    """
        
    assert data in ['MASK','MAT'], \
    '[ERROR] unknow type of data: {}'.format(data)   
        
    im_names = dataset.sets[gt_set]["im_names"]
    timer = Timer()
        
    for im_i in xrange(l_start, l_end):
        im_nm =  im_names[im_i]
        im_out_pn = osp.join(im_out_DIR, im_nm+'.png')
        mat_out_pn = osp.join(mat_out_DIR, im_nm+'.mat')
        _, im_pn = dataset.built_im_path(im_nm, im_in_DIR)
        im_RAW = None    
        timer.tic()
        
        if data == 'MASK' and not osp.exists(im_out_pn):        
            im_RAW = cv2.imread(im_pn)
            im_RAW = cv2.cvtColor(im_RAW, cv2.COLOR_BGR2RGB)
            
            mask = features.SimilarityMask('size' in cfg.SEGM_SS_FEATURE, 
                                           'color' in cfg.SEGM_SS_FEATURE, 
                                           'texture' in cfg.SEGM_SS_FEATURE, 
                                           'fill' in cfg.SEGM_SS_FEATURE)
            
            (R, F, _) = selective_search.hierarchical_segmentation(im_RAW, cfg.SEGM_SS_K, mask)                
            colors = generate_color_table(R)               
            is_mask = False
            
            for depth, label in enumerate(F):
                result = colors[label]
                result = (result*cfg.SEGM_SS_ALPHA + 
                      im_RAW*(1.-cfg.SEGM_SS_ALPHA)).astype(np.uint8)
                
                # extract on targeted depth
                if depth == cfg.SEGM_SS_DEPTH:
                    mask_RAW = result
                    is_mask = True
                    break
                
            assert is_mask == True, "[ERROR] unable to extract segm on depth {}".format(im_nm)
            
            # save raw-img output
            mask_RAW = np.array(mask_RAW, dtype=np.int8)
            #print 'mask_RAW.shape: {}'.format(mask_RAW.shape) 
            cv2.imwrite(im_out_pn, mask_RAW)
            timer.toc()
            print '[INFO] %d/%d use time (s) %f' % (im_i+1, len(im_names), timer.average_time)
        
        """ parse proposals as a MAT for easy handle """
        if data == 'MAT' and not osp.exists(mat_out_pn):            
            assert osp.exists(im_out_pn), \
            '[ERROR] missing MASK for {}. Please generate MASK first.'.format(im_out_pn) 
            
            mask_RAW = cv2.imread(im_out_pn)
            mask_RAW = cv2.cvtColor(mask_RAW, cv2.COLOR_BGR2RGB)
            
            # convert to gray scale for a fast processing    
            mask_RAW = cv2.cvtColor(mask_RAW, cv2.COLOR_RGB2GRAY)
            uniq_vals = np.unique(mask_RAW)
            
            labels = []
            bboxes = []
            superpixels = np.zeros((mask_RAW.shape[0],mask_RAW.shape[1]),dtype=np.float)      
            num_ids = max(min(len(uniq_vals),255),255)
            labels_ids = np.random.permutation(xrange(1,num_ids+1))            
            idx = 0
            
            for val in uniq_vals:
                [row_ids, col_ids] = np.array(np.where(mask_RAW == val))
                
                if len(row_ids) > 0 and len(row_ids)==len(col_ids):                        
                    y1 = np.min(row_ids)
                    x1 = np.min(col_ids)
                    y2 = np.max(row_ids)
                    x2 = np.max(col_ids)
                    box = np.array([x1, y1, x2, y2])
                    
                    assert idx < num_ids, \
                    '[ERROR] idx out labels range: {} vs. {}'.format(idx,num_ids)
                      
                    #label = int('{}{}{}'.format(i_R, i_G, i_B))
                    label = labels_ids[idx]                         
                    superpixels[row_ids,col_ids] = label
                        
                    labels.append(label)
                    bboxes.append(box)
                    idx += 1
                            
            im_mat = {"labels": labels,
                      "bboxes": bboxes, 
                      "superpixels": superpixels}
            
            # save mat-img output        
            sio.savemat(mat_out_pn, im_mat, do_compression=True)
            timer.toc()
            print '[INFO] %d/%d use time (s) %f' % (im_i+1, len(im_names), timer.average_time)

def generate_with_mcg():
    pass


def generate_with_cob():
    pass


def generate_color_table(R):
    # generate initial color
    colors = np.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors


if __name__ == '__main__':
    args = parse_args()
    
    print('Input args:')
    pprint.pprint(args)
    
    # setup & load configs
    _C = Config(config_pn="config/config.ini")
    cfg = _C.cfg
    
    print('Using config:')
    pprint.pprint(cfg)
    
    print ('[INFO] loading dataset {} to generate data...'.format(args["dataset"]))
    dataset_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])
    cls_file = osp.join(cfg.MAIN_DIR_ROOT, "config", "classes.lst")
    
    dataset = None
    ds_pascal = ["demo", "voc_2007", "voc_2012", "bsd_voc2012"]
    
    if args["dataset"] in ds_pascal:
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
    assert dataset != None, \
    "[ERROR] unable to build {} dataset. Available: {}".format(args["dataset"], ds_pascal)
    
    dataset.load_sets()
    print ('[INFO] dataset.name: {}'.format(dataset.name))
    print ('[INFO] dataset.num_cls: {}'.format(dataset.num_cls))
    print ('[INFO] dataset.train: {}'.format(dataset.sets["train"]["num_items"]))
    print ('[INFO] dataset.trainval: {}'.format(dataset.sets["trainval"]["num_items"]))
    print ('[INFO] dataset.test: {}'.format(dataset.sets["test"]["num_items"]))
    print ('[INFO] dataset.val: {}'.format(dataset.sets["val"]["num_items"]))
        
    if args["dataset"] == "bsd_voc2012":
        im_in_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", dataset.name, 'img')
    else:
        im_in_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", dataset.name, 
                                  cfg.PASCAL_DATASET_DIR_IMAGE)
        
    im_out_DIR = args['im_out_dir']
    mat_out_DIR = args['mat_out_dir']
    gt_set = args['gt_set']
    
    # prepare process with multi-process
    processes = []
    num_proc = args['num_proc']
    l_start = 0
    
    if gt_set in dataset.sets.keys():
        im_names = dataset.sets[gt_set]["im_names"]
        num_imgs = dataset.sets[gt_set]["num_items"]
        
        random.shuffle(im_names)
        dataset.sets[gt_set]["im_names"] = im_names
        l_offset = int(np.ceil(num_imgs / float(num_proc)))
    else:
        raise Exception('[ERROR] unknown gt_set defined {}'.format(gt_set))
    
    assert num_imgs !=-1, '[ERROR] invalid number of images: {}'.format(num_imgs)
    
    if not osp.exists(im_out_DIR):
        os.mkdir(im_out_DIR)
    if not osp.exists(mat_out_DIR):
        os.mkdir(mat_out_DIR)
    
    for proc_id in xrange(num_proc):
        l_end = min(l_start + l_offset, num_imgs)
        
        if args['pr_meth'] == 'selective_search':
            if args['mode'] == 'fast':
                p = Process(target=generate_with_selective_search_1d, 
                            args=(l_start, l_end, dataset, gt_set, im_in_DIR, 
                                  im_out_DIR, mat_out_DIR, args['data'], cfg))
            elif args['mode'] == 'accurate':
                p = Process(target=generate_with_selective_search_3d, 
                            args=(l_start, l_end, dataset, gt_set, im_in_DIR, 
                                  im_out_DIR, mat_out_DIR, args['data'], cfg))
            else:
                raise Exception('[ERROR] unknow processing mode: {}'.format(args['mode']))
        elif args['pr_meth'] == 'mcg':
            raise NotImplementedError
        elif args['pr_meth'] == 'cob':
            raise NotImplementedError
        else:
            raise Exception('[ERROR] unknown proposal method {}'.format(args['pr_meth']))
        
        p.start()
        processes.append(p)
        l_start += l_offset
        
    for p in processes:
        p.join()
    
    