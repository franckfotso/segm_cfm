# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: segm_cfm
# Module: tools.prepare_data
# Written by: Romuald FOTSO
# Licensed under MIT License
# Date: 19 Jul 2017
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: python tools/prepare_data.py --dataset voc_2007 \
#        --gt_set train --in_gt_dir data/sbd_voc2007/SEGM-RAW \
#        --in_pr_dir data/sbd_voc2007/SS-RAW --pr_meth selective_search \
#        --num_proc 1 --output_dir cache/sbd_voc2007_train_ss
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import _init_paths
import argparse, pprint, os, gc, resource
from multiprocessing import Process, Lock

from Config import Config
from datasets.CFMGenerator import CFMGenerator
from datasets.Pascal import Pascal
from datasets.Image import Image
from datasets.Mask import Mask
from utils.timer import Timer

import numpy as np
import numpy.random as npr
import os.path as osp
import scipy.io as sio


def parse_args():
    """
        construct the argument parse and parse the arguments
    """    
    ap = argparse.ArgumentParser(description='Prepare data for training')
    ap.add_argument('--dataset', dest='dataset', required=True,
                        help='dataset name to use',
                        default='voc_2007', type=str)
    ap.add_argument('--gt_set', dest='gt_set',
                        help='gt set use to list data',
                        default='train', type=str)
    ap.add_argument('--in_gt_dir', dest='in_gt_dir', required=True,
                        help='input directory gt MAT',
                        default='data/sbd_voc2007/SEGM-RAW', type=str)
    ap.add_argument('--in_pr_dir', dest='in_pr_dir', required=True,
                        help='input directory proposals MAT',
                        default='data/sbd_voc2007/SS-RAW', type=str)    
    ap.add_argument('--pr_meth', dest='pr_meth', 
                        help='proposal method to use', required=True,
                        default='selective_search', type=str)
    ap.add_argument('--top_k', dest='top_k',
                        help='top K proposals to select',
                        default=-1, type=int)
    ap.add_argument('--num_proc', dest='num_proc', 
                        help='numbers of concurrent processes',
                        default=1, type=int)
    ap.add_argument('--output_dir', dest='output_dir', required=True,
                        help='output directory for MAT results',
                        default='cache/sbd_voc2007_train_ss', type=str)
    ap.add_argument('--out_check', dest='out_check',
                        help='cross check integrity of existing outputs',
                        default=False, type=bool)
    return vars(ap.parse_args())


def prepare_data(l_start, l_end, all_data, data_gen, pr_meth, 
                 random_scale_inds, out_check, cfg):
    
    timer = Timer()
    gt_mat_pns, pr_mat_pns, images_pns, outputs_pns = all_data
    
    for im_i in xrange(l_start, l_end):
        gc.collect()
        
        gt_mat_pn = gt_mat_pns[im_i]
        pr_mat_pn = pr_mat_pns[im_i]
        image_pn = images_pns[im_i]
        out_mat_pn = outputs_pns[im_i]
        out_mat_fn = out_mat_pn.split('/')[0]
                
        splts = out_mat_fn.split('.')[0].split('_')
        is_flipped = False
        if splts[-1] == 'flipped':
            is_flipped = True
                        
        timer.tic()
        if osp.exists(out_mat_pn):
            if not out_check:
                continue
            
            is_corrupted = False
            try:
                img_mat = sio.loadmat(out_mat_pn)
                
                tgt_keys = ['sample_rois','im_blob','sample_masks','im_scales']
                for key in tgt_keys:
                    if key not in img_mat.keys():
                        is_corrupted = True                
                del img_mat
                
            except Exception as e:
                print ("[ERROR] exception found: {}".format(e))
                is_corrupted = True
                os.remove(out_mat_pn)
            
            if not is_corrupted:
                timer.toc()
                print '[INFO] %d/%d checked time (s) %f' % (im_i, len(outputs_pns), timer.average_time)
                continue            
        
        # init obj with mat data
        assert osp.exists(gt_mat_pn) and osp.exists(pr_mat_pn), \
        '[ERROR] missing gt & pr mat file required to build training data.'+\
        ' \n gt_mat_pn: {} \n pr_mat_pn: {}'.format(gt_mat_pn,pr_mat_pn)
        
        try:
            gt_mat = sio.loadmat(gt_mat_pn)
            pr_mat = sio.loadmat(pr_mat_pn)
        except Exception as e:
            print ("[INFO] pr_mat_pn: {}".format(pr_mat_pn))
            raise Exception("[ERROR] exception found: {}".format(e))
        
        # split superpixels into mask instances
        inst_ids = gt_mat['inst_ids'][0]
        gt_superpixels = gt_mat['superpixels']
        gt_labels = gt_mat['labels'][0]
        gt_mask_instances = np.zeros((len(gt_labels),gt_superpixels.shape[0],
                                      gt_superpixels.shape[1]), dtype=np.float)      
        for m_i, (label, inst_id) in enumerate(zip(gt_labels, inst_ids)):
            [row_ids, col_ids] = np.array(np.where((gt_superpixels == inst_id)))
            gt_mask_instances[m_i,row_ids, col_ids] = 1
           
        im_info = {"width": gt_superpixels.shape[1],
                   "height": gt_superpixels.shape[0],
                   "depth": 1}
        
        mask = Mask(filename=gt_mat_pn.split('/')[0],pathname=gt_mat_pn)
        data = {'labels': gt_labels,
                'bboxes': gt_mat['bboxes'],
                'im_info': im_info,
                'flipped': is_flipped,
                'instances': gt_mask_instances.copy()
                }
        mask.gt_data = data
                
        image = Image(filename=image_pn.split('/')[0],pathname=image_pn)
        rois = {
                'boxes': gt_mat['bboxes'],
                'im_info': im_info,
                'flipped': is_flipped
                }
        image.gt_rois = rois        
        
        del gt_mask_instances
        del gt_superpixels
        del inst_ids
        
        if pr_meth == "selective_search":
            pr_labels = pr_mat['labels'][0]
            pr_superpixels = pr_mat['superpixels']
            
        elif pr_meth == "mcg":
            pr_labels = pr_mat['labels'][0]
            pr_bboxes = pr_mat['bboxes']
            pr_superpixels = pr_mat['superpixels']            
            
        elif pr_meth == "cob":
            pr_labels = pr_mat['labels'][0]
            pr_bboxes = pr_mat['bboxes']
            pr_superpixels = pr_mat['superpixels']
            
        else:
            raise NotImplemented
        #print 'pr_labels.shape: {}'.format(pr_labels.shape)
        #print 'pr_superpixels.shape: {}'.format(pr_superpixels.shape)
        #print 'len(pr_labels): {}'.format(len(pr_labels))
        #raise Exception('pause')
        
        if len(pr_superpixels.shape) == 2:
            pr_mask_instances = np.zeros((len(pr_labels),pr_superpixels.shape[0],
                                          pr_superpixels.shape[1]), dtype=np.float)
            pr_bboxes = np.zeros((len(pr_labels),4))
            
            for m_i, label in enumerate(pr_labels):
                tgt_ids = np.array(np.where((pr_superpixels == label)))
                if len(tgt_ids) == 2:
                    #print 'tgt_ids.shape: {}'.format(tgt_ids.shape)
                    #print 'tgt_ids: {}'.format(tgt_ids)
                    [row_ids, col_ids] = tgt_ids
                    pr_mask_instances[m_i, row_ids, col_ids] = 1                
                    y1 = np.min(row_ids)
                    x1 = np.min(col_ids)
                    y2 = np.max(row_ids)
                    x2 = np.max(col_ids)
                    pr_bboxes[m_i] = np.array([x1, y1, x2, y2])
        elif len(pr_superpixels.shape) == 3:
            pr_mask_instances = pr_superpixels
        else:
            raise Exception('[ERROR] incorrect shape for pr_superpixels: {}'.format(pr_superpixels.shape))
            
        data = {'labels': pr_labels,
                'bboxes': pr_bboxes,
                'im_info': im_info,
                'flipped': is_flipped,
                'instances': pr_mask_instances.copy()
                }
        mask.pr_data = data
        
        rois = {
                'boxes': pr_mat['bboxes'],
                'im_info': im_info,
                'flipped': is_flipped
                }
        image.pr_rois = rois
        
        del pr_mask_instances
        del pr_labels
        del pr_superpixels
        del gt_mat
        del pr_mat
        
        #lock.acquire()
        # resize & build data blob
        im_blob, im_scales = data_gen.built_image_blob(image, random_scale_inds[im_i])
        
        # build RAW GT & PR mask_instances ==> WARN: seems to be highly memory consuming
        #all_mask_RAWs, all_mask_scales = data_gen.built_all_masks(mask, random_scale_inds[im_i])
        
        # group gt & pr masks
        #print "mask.gt_data['instances'].shape: {}".format(mask.gt_data['instances'].shape)
        #print "mask.pr_data['instances'].shape: {}".format(mask.pr_data['instances'].shape)
        num_gt, mask_h, mask_w = mask.gt_data['instances'].shape
        num_pr = mask.pr_data['instances'].shape[0]
        num_masks = num_gt+num_pr
        all_mask_RAWs = np.zeros((num_masks, mask_h, mask_w))
        #print "all_mask_RAWs.shape: {}".format(all_mask_RAWs.shape)
        all_mask_RAWs[0:num_gt] = mask.gt_data['instances']
        all_mask_RAWs[num_gt:num_masks] = mask.pr_data['instances']
        del mask
        
        gt_boxes = image.gt_rois['boxes']
        pr_boxes = image.pr_rois['boxes']
        
        gt_classes = gt_labels
        gt_classes = np.reshape(gt_classes, (gt_classes.shape[0],1)) 
        #print 'gt_boxes.shape: {}'.format(gt_boxes.shape)
        #print 'gt_classes.shape: {}'.format(gt_classes.shape)
        
        gt_zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        gt_boxes = np.hstack((gt_boxes, gt_classes)) # ( x1, y1, x2, y2, cls )       
        pr_zeros = np.zeros((pr_boxes.shape[0], 1), dtype=pr_boxes.dtype)
        pr_boxes = np.hstack((pr_zeros, pr_boxes))            
        all_rois = np.vstack((np.hstack((gt_zeros, gt_boxes[:, :-1])), pr_boxes))       
        
        #print 'gt_boxes.shape: {}'.format(gt_boxes.shape)
        #print 'all_rois.shape: {}'.format(all_rois.shape)       
        #print 'gt_boxes[0]: {}'.format(gt_boxes[0])
        #print 'all_rois[0]: {}'.format(all_rois[0])        
        
        sample_rois, sample_masks = data_gen.sample_rois_masks(all_rois, gt_boxes, dataset.num_cls, \
                                                               image, all_mask_RAWs, im_scales, cfg)
                
        cache_im_data = {
                        'im_blob': im_blob,
                        'im_scales': im_scales,
                        'sample_rois': sample_rois,
                        'sample_masks': sample_masks
                        }                
        try:
            sio.savemat(out_mat_pn, cache_im_data, do_compression=True)
        except Exception as e:
            print ("[WARM] exception found: {}, trying sio.savemat without compression".format(e))
            sio.savemat(out_mat_pn, cache_im_data, do_compression=False)
        except Exception as f:            
            print ("[ERROR] exception found: {}".format(f))

        del sample_rois
        del sample_masks
        gc.collect()
        timer.toc()
        print '[INFO] {}/{} use time (s) {:.3f}'.format(im_i, len(outputs_pns), timer.average_time)
        
def prepare_mcg_data(l_start, l_end, data_gen, gt_mat_dir, pr_mat_dir, output_dir, cfg):
    pass


def prepare_cob_data():
    pass


if __name__ == '__main__':
    """
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    print 'Soft limit starts as  :', soft
    resource.setrlimit(rsrc, (40000000000, hard)) #limit to 40G
    soft, hard = resource.getrlimit(rsrc)
    print 'Soft limit changed to :', soft
    """
    args = parse_args()
    
    print('Input args:')
    pprint.pprint(args)
    
    # setup & load configs
    _C = Config(config_pn="config/config.ini")
    cfg = _C.cfg
    
    print('Using config:')
    pprint.pprint(cfg)
    
    print ('[INFO] loading dataset {} to prepare data...'.format(args["dataset"]))
    dataset_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])
    cls_file = osp.join(cfg.MAIN_DIR_ROOT, "config", "classes.lst")
    
    dataset = None
    ds_pascal = ["voc_2007", "voc_2012", "bsd_voc2012"]
    
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
    
    im_in_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", dataset.name, 
                                  cfg.PASCAL_DATASET_DIR_IMAGE)
    in_gt_dir = args['in_gt_dir']
    in_pr_dir = args['in_pr_dir']
    output_dir = args['output_dir']
    gt_set = args['gt_set']
    pr_meth = args['pr_meth']
    out_check = args['out_check']
    num_proc = args['num_proc']
    top_k = args['top_k']
    
    gt_masks, gt_images = dataset.load_gt_masks_rois(gt_set=gt_set, in_gt_dir=in_gt_dir, num_proc=num_proc)
    pr_masks, pr_images = dataset.load_mask_rois_proposals(gt_set=gt_set, in_pr_dir=in_pr_dir,
                                                                        src=pr_meth, top_k=top_k, num_proc=num_proc)
    print ('[INFO] gt_images.num: {}, gt_masks.num: {}'.format(len(gt_images),len(gt_masks)))
    print ('[INFO] pr_images.num: {}, pr_masks.num: {}'.format(len(pr_images),len(pr_masks)))
    
    all_images = dataset.merge_gt_proposal_rois(gt_images, pr_images)
    all_masks = dataset.merge_gt_proposal_masks(gt_masks, pr_masks)
    print ('[INFO] all_images.num: {}'.format(len(all_images)))
    print ('[INFO] all_masks.num: {}'.format(len(all_masks)))
    
    gt_mat_pns = []
    pr_mat_pns = []
    images_pns = []
    outputs_pns = []
    for image in all_images:
        im_nm = (image.filename).split('.')[0]
        if image.gt_rois['flipped']:
            gt_mat_pn = osp.join(in_gt_dir, im_nm+'_flipped.mat')
            pr_mat_pn = osp.join(in_pr_dir, im_nm+'_flipped.mat')
            out_mat_pn = osp.join(output_dir, im_nm+'_flipped.mat')
        else:
            gt_mat_pn = osp.join(in_gt_dir, im_nm+'.mat')
            pr_mat_pn = osp.join(in_pr_dir, im_nm+'.mat')
            out_mat_pn = osp.join(output_dir, im_nm+'.mat')
            
        gt_mat_pns.append(gt_mat_pn)
        pr_mat_pns.append(pr_mat_pn)
        images_pns.append(image.pathname)
        outputs_pns.append(out_mat_pn)
      
    all_data = [gt_mat_pns, pr_mat_pns, images_pns, outputs_pns]
    
    num_imgs = len(gt_mat_pns)
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN_DEFAULT_SCALES),
                                    size=num_imgs)
    
    # prepare process with multi-process
    processes = []    
    l_start = 0   
    l_offset = int(np.ceil(num_imgs / float(num_proc)))
    
    data_gen = CFMGenerator(images=all_images, masks=None, 
                            num_cls=dataset.num_cls, 
                            bbox_means=None, bbox_stds=None,
                            in_gt_dir=None, in_pr_dir=in_pr_dir,
                            cfm_t=None, cache_im_dir=output_dir, cfg=cfg)
    del data_gen.images
    del all_masks
    del all_images
                            
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
    print ('[INFO] preparing data with the proposal method: {}'.format(pr_meth))
    for proc_id in xrange(num_proc):
        l_end = min(l_start + l_offset, num_imgs)
        
        p = Process(target=prepare_data, 
                    args=(l_start, l_end, all_data, data_gen, 
                            pr_meth, random_scale_inds, out_check, cfg))        
        p.start()
        processes.append(p)
        l_start += l_offset
        
    for p in processes:
        if p.exitcode != None:
            p.terminate()
            
        p.join()
    
    
