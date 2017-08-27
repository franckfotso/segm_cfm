# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: segm_cfm
# Module: tools.demo
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: python tools/train_net.py --gpu_id 0 --dataset voc_2007 \
#        --gt_set train --solver models/AlexNet/solver_CFM_T1.prototxt \
#        --weights pretrained/AlexNet/bvlc_alexnet.caffemodel \
#        --iters 30000 --cache_im_dir cache/sbd_voc2007_train_ss
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import print_function
import _init_paths
import argparse, pprint, caffe
import os.path as osp
import numpy as np

from Config import Config
from datasets.Pascal import Pascal
from main.Trainer import Trainer

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", dest="gpu_id", default=0, 
                    help="id of GPU device", type=int)
    ap.add_argument("--dataset", dest="dataset", required=True, 
                    help="dataset name to use", type=str)
    ap.add_argument('--gt_set', dest='gt_set',
                    help='gt set use to list data', required=True,
                    default='train', type=str)
    ap.add_argument('--in_gt_dir', dest='in_gt_dir', required=True,
                        help='input directory gt MAT',
                        default='data/sbd_voc2007/SEGM-RAW', type=str)
    ap.add_argument('--in_pr_dir', dest='in_pr_dir', required=True,
                        help='input directory proposals MAT',
                        default='data/sbd_voc2007/SS-RAW', type=str)
    ap.add_argument('--task', dest='task',
                    help='type of task to execute', required=True,
                    default='CFM', type=str)
    ap.add_argument("--solver", dest="solver", required=True, 
                    help="solver required for training", type=str)
    ap.add_argument("--weights", dest="weights", required=True, 
                    help="initialize with pretrained model weights", type=str)
    ap.add_argument("--iters", dest="iters", default=40000,
                    help="number of iterations to train", type=int)
    ap.add_argument('--cache_im_dir', dest='cache_im_dir', required=True,
                    help='directory for images prepared', type=str)
    ap.add_argument("--rand", dest="rand", default=False,
                    help="randomize (do not use a fixed seed)", type=bool)
    ap.add_argument("--verbose", dest="verbose", default=False,
                    help="verbosity level", type=bool)
    args = vars(ap.parse_args())
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    print('[INFO] Called with args:')
    print(args)
    
    # setup & load configs
    _C = Config(config_pn="config/config.ini")
    cfg = _C.cfg
    
    cfg.MAIN_DEFAULT_GPU_ID = args['gpu_id']
    
    print('Using config:')
    pprint.pprint(cfg)
    
    if not args['rand']:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.MAIN_DEFAULT_RNG_SEED)
        caffe.set_random_seed(cfg.MAIN_DEFAULT_RNG_SEED)
        
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args['gpu_id'])
    
    print ('[INFO] loading dataset {} for training...'.format(args["dataset"]))
    dataset_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])
    gt_set = args['gt_set']
    task = args['task']
    in_gt_dir = args['in_gt_dir']
    in_pr_dir = args['in_pr_dir']
    
    dataset = None
    ds_pascal = ["voc_2007", "bsd_voc2012"]
    
    if args["dataset"] in  ds_pascal:
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
        
    gt_masks, gt_images = dataset.load_gt_masks_rois(gt_set=gt_set, in_gt_dir=in_gt_dir)
    pr_masks, pr_images = dataset.load_mask_rois_proposals(gt_set=gt_set, in_pr_dir=in_pr_dir, 
                                                           src=cfg.TRAIN_DEFAULT_SEGM_METHOD)
    print ('[INFO] gt_images.num: {}, gt_masks.num: {}'.format(len(gt_images),len(gt_masks)))
    print ('[INFO] pr_images.num: {}, pr_masks.num: {}'.format(len(pr_images),len(pr_masks)))
    
    all_images = dataset.merge_gt_proposal_rois(gt_images, pr_images)
    all_masks = dataset.merge_gt_proposal_masks(gt_masks, pr_masks)
    print ('[INFO] all_images.num: {}'.format(len(all_images)))
    print ('[INFO] all_masks.num: {}'.format(len(all_masks)))
    
    model = {'solver':args['solver'],'pretrained_model': args['weights']}
    trainer = Trainer(dataset=dataset, model=model, cfg=cfg)
    
    # launch train process
    root_dir = cfg.MAIN_DIR_ROOT
    output_dir = dataset.built_output_dir(root_dir, 'train')
    cache_im_dir = osp.join(root_dir, args['cache_im_dir'])
    print ('[INFO] Start the training process on {}'.format(args["dataset"]))
    trainer.run(all_images, all_masks, cache_im_dir, output_dir, task, max_iters=args['iters'])


