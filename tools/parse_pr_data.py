# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: segm_cfm
# Module: tools.parse_segm_data
# Copyright (c) 2017
# Written by: Romuald FOTSO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: python tools/parse_pr_data.py --dataset voc_2007 \
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
    ap.add_argument('--pr_meth', dest='pr_meth', required=True,
                        help='proposal method',
                        default='mcg', type=str)
    ap.add_argument('--in_pr_dir', dest='in_pr_dir', required=True,
                        help='input proposals mat directory',
                        default='data/bsd_voc_2012/MCG-RAW.init', type=str)
    ap.add_argument('--num_proc', dest='num_proc', 
                        help='numbers of concurrent processes',
                        default=1, type=int)
    ap.add_argument('--top_k', dest='top_k', 
                        help='top k proposals to parse',
                        default=-1, type=int)
    ap.add_argument('--out_pr_dir', dest='out_pr_dir', required=True,
                        help='output proposals mat directory',
                        default='data/bsd_voc_2012/MCG-RAW.parsed', type=str)
    return vars(ap.parse_args())
    

def parse_from_mcg(l_start, l_end, dataset, gt_set, top_k, in_pr_dir, out_pr_dir):                          
    im_names = dataset.sets[gt_set]["im_names"]
    timer = Timer()
    
    for im_i in xrange(l_start, l_end):
        im_nm =  im_names[im_i]        
        pr_mat_pn = osp.join(in_pr_dir, im_nm+'.mat')
        out_mat_pn = osp.join(out_pr_dir, im_nm+'.mat')
        timer.tic()
        
        if osp.exists(out_mat_pn):
            continue        
        
        assert osp.exists(pr_mat_pn), \
        '[ERROR] unable to find proposal MAT file: {}, d'+ \
        'did you call tools/generate_data.py on your dataset ?'.format(pr_mat_pn)
        
        pr_mat = sio.loadmat(pr_mat_pn)
        mcg_labels = pr_mat['labels']
        mcg_scores = pr_mat['scores']
        mcg_superpixels = pr_mat['superpixels']     
        
        bboxes = []
        labels = []       
        # keep only proposals with score > 0: rel. mat size issue
        scores_ids = np.where(np.reshape(mcg_scores, (-1,)) > 0)[0]
        mcg_labels = mcg_labels[scores_ids]
        num_pr = len(mcg_labels)
        if top_k != -1  and num_pr > top_k:
            num_pr = top_k
        superpixels = np.zeros(((num_pr,)+mcg_superpixels.shape))   
        
        for i_lab in xrange(num_pr):
            label = mcg_labels[i_lab][0][0]
            proposal = np.in1d(mcg_superpixels, label).reshape(mcg_superpixels.shape)
            [row_ids, col_ids] = np.where(proposal == 1)
            
            if len(row_ids) > 0 and len(row_ids)==len(col_ids):             
                superpixels[i_lab, row_ids, col_ids] = 1
                y1 = np.min(row_ids)
                x1 = np.min(col_ids)
                y2 = np.max(row_ids)
                x2 = np.max(col_ids)
                box = np.array([x1, y1, x2, y2])
                bboxes.append(box)
                labels.append(i_lab+1)
        
        parsed_mat = {"labels": np.array(labels),
                      "bboxes": np.array(bboxes),
                      "superpixels": superpixels}
                    
        # save mat-mask output
        try:
            sio.savemat(out_mat_pn, parsed_mat, do_compression=True)
        except Exception as e:
            print ("[WARM] exception found: {}, trying sio.savemat without compression".format(e))
            sio.savemat(out_mat_pn, parsed_mat, do_compression=False)
        except Exception as f:            
            print ("[ERROR] exception found: {}".format(f))
        timer.toc()
        print '[INFO] %d/%d use time (s) %f' % (im_i, len(im_names), timer.average_time)
        
        
def parse_from_cob(l_start, l_end, dataset, gt_set,
                          max_pr, in_seg_dirs, out_seg_dir):                              
    pass
        
        
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
    
    in_pr_dir = args['in_pr_dir']    
    out_pr_dir = args['out_pr_dir']
    gt_set = args['gt_set']
    pr_meth = args['pr_meth']
    top_k = args['top_k']
    
    # prepare process with multi-process
    processes = []
    num_proc = args['num_proc']
    l_start = 0
    
    assert gt_set in dataset.sets.keys(), \
    '[ERROR] unknown gt_set defined {}'.format(gt_set)
    
    im_names = dataset.sets[gt_set]["im_names"]
    num_imgs = dataset.sets[gt_set]["num_items"]
    l_offset = int(np.ceil(num_imgs / float(num_proc)))
    
    if not osp.exists(out_pr_dir):
        os.mkdir(out_pr_dir)
    
    for proc_id in xrange(num_proc):
        l_end = min(l_start + l_offset, num_imgs)
        
        if pr_meth == "mcg":
            p = Process(target=parse_from_mcg, 
                            args=(l_start, l_end, dataset, gt_set,
                                  top_k, in_pr_dir, out_pr_dir))
        elif pr_meth == "cob":
            p = Process(target=parse_from_cob, 
                            args=(l_start, l_end, dataset, gt_set,
                                  top_k, in_pr_dir, out_pr_dir))
        else:
            raise NotImplementedError
        
        p.start()
        processes.append(p)
        l_start += l_offset
        
    for p in processes:
        p.join()    
    