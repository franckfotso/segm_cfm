# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: segm_cfm
# Module: tools.demo
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: 
# python tools/demo.py --gpu_id 0 --net AlexNet \
#  --gt_set test --task CFM --dataset demo \
#  --proto models/AlexNet/AlexNet_CFM_T1_test.prototxt \
#  --weights outputs/voc_2007/train/AlexNet_segm_cfm_T1_iter_500.caffemodel \
#  --cache_im_dir cache/sbd_demo_test_ss --kwargs 'cfm_t:T1' 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import print_function
import _init_paths
import argparse, pprint, caffe, cv2, os
from Config import Config
from PIL import Image
from datasets.CFMGenerator import CFMGenerator
from datasets.Pascal import Pascal
from main.TestWrapper import TestWrapper
from utils.timer import Timer
from utils.transformation import *
from utils.vis_seg import _convert_pred_to_image, _get_voc_color_map

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


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
    ap.add_argument('--in_pr_dir', dest='in_pr_dir', required=True,
                        help='input directory proposals MAT', type=str) 
    ap.add_argument("--rand", dest="rand", default=False,
                    help="randomize (do not use a fixed seed)", type=bool)
    ap.add_argument("--verbose", dest="verbose", default=False,
                    help="verbosity level", type=bool)
    ap.add_argument("--kwargs", dest="kwargs", default=None,
                    help="(optional) further args ", type=str)
    args = vars(ap.parse_args())
    
    return args

def get_vis_dict(result_box, result_mask, img_name, cls_names, vis_thresh=0.5):
    box_for_img = []
    mask_for_img = []
    cls_for_img = []
    for cls_ind, _ in enumerate(cls_names):
        det_for_img = result_box[cls_ind]
        seg_for_img = result_mask[cls_ind]
        keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
        for keep in keep_inds:
            box_for_img.append(det_for_img[keep])
            mask_for_img.append(seg_for_img[keep])
            cls_for_img.append(cls_ind + 1)
    res_dict = {'image_name': img_name,
                'cls_name': cls_for_img,
                'boxes': box_for_img,
                'masks': mask_for_img}
    return res_dict

def vis_detections(im, class_name, dets):
    """Draw detected bounding boxes."""
    #print ('dets.shape: {}'.format(dets.shape))
    #print ('inds.shape: {}'.format(inds.shape))
        
    l_bboxes = []
    n_dets = dets.shape[0]
    for i in xrange(n_dets):
        bbox = dets[i, :4].astype(np.int)
        score = dets[i, -1]
        
        # add by rfm
        print ('Det: (x_min,y_min,W,H) = ({},{},{},{}), class_name = {:s}, score = {:.3f}'.\
        format(bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1],class_name,score))
        
        l_bboxes.append({'x_min':bbox[0],'y_min':bbox[1],\
                         'x_max':bbox[2],'y_max':bbox[3],\
                         'cls':class_name,'score':score})

    return l_bboxes 

def demo_model(gt_set, tw, image, backbone_net, task, kwargs=None):
    if task == 'CFM':
        assert 'cfm_t' in kwargs.keys(), \
        '[ERROR] missing type of CFM to process'
        
        cfm_t = kwargs['cfm_t']        
        outputs = tw.cfm_net_forward(cfm_t, image, backbone_net)
        
    elif task == 'MNC':
        raise NotImplemented    
    elif task == 'FCIS':
        raise NotImplemented    
    elif task == 'MRCNN':
        raise NotImplemented
    
    return outputs


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
    in_pr_dir = args['in_pr_dir']
    
    dataset = None
    ds_names = ["demo", "voc_2007"]
    
    if args["dataset"] == "voc_2007":
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
        
    elif args["dataset"] == "demo":
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
        
    assert dataset != None, \
    "[ERROR] unable to build {} dataset. Available: {}".format(args["dataset"], ds_names)
    
    dataset.load_sets()
    print ('[INFO] dataset.name: {}'.format(dataset.name))
    print ('[INFO] dataset.num_cls: {}'.format(dataset.num_cls))
    print ('[INFO] dataset.train: {}'.format(dataset.sets["train"]["num_items"]))
    print ('[INFO] dataset.trainval: {}'.format(dataset.sets["trainval"]["num_items"]))
    print ('[INFO] dataset.test: {}'.format(dataset.sets["test"]["num_items"]))
    print ('[INFO] dataset.val: {}'.format(dataset.sets["val"]["num_items"]))
        
    all_masks, all_images = dataset.load_mask_rois_proposals(gt_set=gt_set, 
                                                            in_pr_dir=in_pr_dir, 
                                                            src=cfg.TEST_DEFAULT_SEGM_METHOD)
    print ('[INFO] all_images.num: {}'.format(len(all_images)))
    print ('[INFO] all_masks.num: {}'.format(len(all_masks)))
    
    model = {'proto':args['proto'],'weights': args['weights']}
    
    # launch train process
    output_dir = dataset.built_output_dir(root_dir=cfg.MAIN_DIR_ROOT, phase=gt_set)
    #cache_im_dir = osp.join(root_dir, args['cache_im_dir'])
    
    assert task in cfg.MAIN_DEFAULT_TASKS, \
            '[ERROR] unknown task name provided: {}'.format(task)
            
    # disable the shuffling in the test set
    cfg.TRAIN_DEFAULT_ASPECT_GROUPING = False 
        
    if task == 'CFM':        
        data_gen = CFMGenerator(images=all_images, masks=all_masks,
                                bbox_means=None, bbox_stds=None,                                
                                num_cls=dataset.num_cls, cfm_t=None,
                                in_gt_dir=None, in_pr_dir=in_pr_dir,
                                cache_im_dir=None, cfg=cfg)                    
    elif task == 'MNC':
        raise NotImplemented    
    elif task == 'FCIS':
        raise NotImplemented    
    elif task == 'MRCNN':
        raise NotImplemented
    
    tw = TestWrapper(args['proto'], 
                     args['weights'], task,
                     data_gen, output_dir, cfg)
    
    timer = Timer()
    
    for im_i in xrange(len(all_images)):
        image = all_images[im_i]
        
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Demo - processing: {}'.format(image.filename))
        
        timer.tic()
        outputs = demo_model(gt_set, tw, image, backbone_net, task, kwargs)
        timer.toc()
        print ('forward time (s) %f' % (timer.average_time))
        out_boxes = outputs['final_boxes']
        out_masks = outputs['final_masks']
        seg_scores = outputs['final_masks_scores']
        
        if task == 'CFM': 
            # pooled mask bef mask voting: T1 & T2
            assert 'cfm_t' in kwargs.keys(), \
                '[ERROR] missing type of CFM to process'
            
            cfm_t = kwargs['cfm_t']
            out_masks_pooled = out_masks        
        
        img_width = image.pr_rois['im_info']['width']
        img_height = image.pr_rois['im_info']['height']
        
        # apply the mask voting process
        if cfg.TEST_DEFAULT_USE_GPU_MASK_MERGE:
            final_masks, final_bboxes = gpu_mask_voting(out_masks_pooled, out_boxes, 
                                                        seg_scores, dataset.num_cls, 100,
                                                        img_width, img_height, cfg)
        else:
            final_masks, final_bboxes = cpu_mask_voting(out_masks_pooled, out_boxes, 
                                                        seg_scores, dataset.num_cls, 100,
                                                        img_width, img_height, cfg)
        
        im = cv2.imread(image.pathname)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        im_draw = im.copy()
        im_txt = ('img: {}').format(image.pathname)
        
        pred_dict = get_vis_dict(final_bboxes, final_masks, 
                                 image.pathname, dataset.classes[1:], 
                                 vis_thresh=cfg.TEST_DEFAULT_CONF_THRESH)
        
        inst_img, cls_img = _convert_pred_to_image(img_width, img_height, pred_dict, cfg)
        
        color_map = _get_voc_color_map()

        spl_dirs = image.pathname.split('/')
        demo_out_dir = '/'.join(spl_dirs[0:len(spl_dirs)-2])
        demo_out_dir = osp.join(demo_out_dir, "out")
        
        if not osp.exists(demo_out_dir):
            os.mkdir(demo_out_dir)
        
        target_cls_file = os.path.join(demo_out_dir, 'cls_' + image.filename)
        cls_out_img = np.zeros((img_height, img_width, 3))
        for i in xrange(img_height):
            for j in xrange(img_width):
                cls_out_img[i][j] = color_map[cls_img[i][j]][::-1]
        cv2.imwrite(target_cls_file, cls_out_img)
        
        background = Image.open(image.pathname)
        mask = Image.open(target_cls_file)
        background = background.convert('RGBA')
        mask = mask.convert('RGBA')
        superimpose_image = Image.blend(background, mask, 0.8)
        superimpose_name = os.path.join(demo_out_dir, 'final_' + image.filename)
        superimpose_image.save(superimpose_name, 'JPEG')
        im = cv2.imread(superimpose_name)
        
        classes = pred_dict['cls_name']
        for i in xrange(len(classes)):
            score = pred_dict['boxes'][i][-1]
            bbox = (pred_dict['boxes'][i][:4]).astype(np.int)
            cls_ind = classes[i] - 1              
            print ('segm_cfm: (x1,y1,x2,y2) = ({},{},{},{}), class_name = {:s}, score = {:.4f}'.\
            format(bbox[0],bbox[1],bbox[2],bbox[3],dataset.classes[1:][cls_ind],score))
            cv2.putText(im,'{:s}:{:.3f}'.format(dataset.classes[1:][cls_ind], score),
                    (bbox[0], bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)     
        
        im_draw_pn = osp.join(demo_out_dir, image.filename.split('.')[0]+".png")
        cv2.imwrite(im_draw_pn, im)
        
        os.remove(superimpose_name)
        os.remove(target_cls_file)
        
               
    
    