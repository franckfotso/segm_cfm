# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: segm_cfm
# Module: tools.test_net
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: 
# python tools/test_net.py --gpu_id 0 --net AlexNet \
#  --gt_set test --task CFM --dataset voc_2007
#  --proto models/AlexNet/AlexNet_CFM_T1_test.prototxt \
#  --weights outpus/voc_2007/train/AlexNet_segm_cfm_T1_iter_500.caffemodel \
#  --cache_im_dir cache/sbd_voc2007_train_ss --kwargs 'cfm_t:T1' 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import print_function
import _init_paths
import argparse, pprint, caffe
import os.path as osp
import numpy as np

from Config import Config
from datasets.Pascal import Pascal
from main.Tester import Tester

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", dest="gpu_id", default=0, 
                    help="id of GPU device", type=int)
    ap.add_argument("--task", dest="task", required=True, 
                    help="the type of task to perform", 
                    default='CFM', type=str)
    ap.add_argument("--net", dest="net", required=True, 
                    help="the backbone network involved", 
                    default='VGG16', type=str)
    ap.add_argument("--dataset", dest="dataset", required=True, 
                    help="dataset name to use", type=str)
    ap.add_argument('--gt_set', dest='gt_set',
                    help='gt set use to list data',
                    default='test', type=str)
    ap.add_argument("--proto", dest="proto", required=True, 
                    help="prototxt describing the model", type=str)
    ap.add_argument("--weights", dest="weights", required=True, 
                    help="model use for the test process", type=str)
    ap.add_argument('--cache_im_dir', dest='cache_im_dir', required=True,
                    help='directory for images prepared', type=str)
    ap.add_argument("--rand", dest="rand", default=False,
                    help="randomize (do not use a fixed seed)", type=bool)
    ap.add_argument("--verbose", dest="verbose", default=False,
                    help="verbosity level", type=bool)
    ap.add_argument("--kwargs", dest="kwargs", default=None,
                    help="(optional) further args ", type=str)
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
    
    kwargs = {}
    if args['kwargs'] != None:
        for bloc in args['kwargs'].split('.'):
            param, val = bloc.split(':')
            kwargs[param] = val
    
    if not args['rand']:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.MAIN_DEFAULT_RNG_SEED)
        caffe.set_random_seed(cfg.MAIN_DEFAULT_RNG_SEED)
        
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args['gpu_id'])
    
    print ('[INFO] loading dataset {} for training...'.format(args["dataset"]))
    dataset_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])    
    task = args['task']
    gt_set = args['gt_set']
    backbone_net = args['net']
    
    dataset = None
    ds_names = ["voc_2007"]
    
    if args["dataset"] == "voc_2007":
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
        in_gt_dir = 'data/sbd_voc2007/SEGM-RAW'
        in_pr_dir = 'data/sbd_voc2007/SS-RAW'
    assert dataset != None, \
    "[ERROR] unable to build {} dataset. Available: {}".format(args["dataset"], ds_names)
        
    dataset.load_sets()
    print ('[INFO] dataset.name: {}'.format(dataset.name))
    print ('[INFO] dataset.num_cls: {}'.format(dataset.num_cls))
    print ('[INFO] dataset.train: {}'.format(dataset.sets["train"]["num_items"]))
    print ('[INFO] dataset.trainval: {}'.format(dataset.sets["trainval"]["num_items"]))
    print ('[INFO] dataset.test: {}'.format(dataset.sets["test"]["num_items"]))
    print ('[INFO] dataset.val: {}'.format(dataset.sets["val"]["num_items"]))
        
    gt_masks, gt_images = dataset.load_gt_masks_rois(gt_set, in_gt_dir, in_pr_dir)
    pr_masks, pr_images = dataset.load_mask_rois_proposals(gt_set, in_gt_dir, in_pr_dir, 
                                                           src='selective_search')
    print ('[INFO] gt_images.num: {}, gt_masks.num: {}'.format(len(gt_images),len(gt_masks)))
    print ('[INFO] pr_images.num: {}, pr_masks.num: {}'.format(len(pr_images),len(pr_masks)))
    
    all_images = dataset.merge_gt_proposal_rois(gt_images, pr_images)
    all_masks = dataset.merge_gt_proposal_masks(gt_masks, pr_masks)
    print ('[INFO] all_images.num: {}'.format(len(all_images)))
    print ('[INFO] all_masks.num: {}'.format(len(all_masks)))
    
    model = {'proto':args['proto'],'weights': args['weights']}
    tester = Tester(dataset=dataset, model=model, task=args['task'], cfg=cfg)
    
    # launch train process
    root_dir = osp.join(osp.dirname(__file__), '..')
    output_dir = dataset.built_output_dir(root_dir, gt_set)
    cache_im_dir = osp.join(root_dir, args['cache_im_dir'])
    
    print ('[INFO] Start the test process on {}'.format(args["dataset"]))
    tester.run(gt_set, all_images, all_masks, backbone_net, 
               task, cache_im_dir, output_dir, kwargs)


