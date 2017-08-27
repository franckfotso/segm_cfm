# Project: segm_cfm
# Module: libs.SolverWrapper
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Based on: py-faster-rcnn 
#    (https://github.com/rbgirshick/py-faster-rcnn)
# Licensed under MIT License

import caffe, os
import os.path as osp
import numpy as np

from utils.timer import Timer
from datasets.CFMGenerator import CFMGenerator
from utils.regression import *

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, pretrained_model,
                 data_gen, output_dir, cfg):
        """Initialize the SolverWrapper."""
        self.data_gen = data_gen
        self.output_dir = output_dir
        self.cfg = cfg
        
        if cfg.TRAIN_DEFAULT_BBOX_REG:
            cache_dir = osp.join(cfg.MAIN_DIR_ROOT,cfg.MAIN_DIR_CACHE)
            means_file = osp.join(cache_dir,'{}_bbox_means.npy'\
                                   .format(cfg.TRAIN_DEFAULT_SEGM_METHOD))
            stds_file = osp.join(cache_dir, '{}_bbox_stds.npy'\
                                   .format(cfg.TRAIN_DEFAULT_SEGM_METHOD))
            #print 'means_file: {}'.format(means_file)
            #print 'stds_file: {}'.format(stds_file)
            
            if os.path.exists(means_file) and os.path.exists(stds_file):
                self.bbox_means = np.load(means_file)
                self.bbox_stds = np.load(stds_file)
            else:
                print ('[INFO] SolverWrapper: compute bbox means & stds over the train set...')
                self.bbox_means, self.bbox_stds = compute_bbox_means_stds(data_gen.cache_im_dir, 
                                                                          data_gen.num_cls, cfg)                                                                
            self.data_gen.bbox_means = self.bbox_means
            self.data_gen.bbox_stds = self.bbox_stds
            

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        # set the data_generator in InputData layer
        self.solver.net.layers[0].set_data_generator(self.data_gen) 
        

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net
        scale_bbox_params = (self.cfg.TRAIN_DEFAULT_BBOX_REG and
                             self.cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_TARGETS and
                             'bbox_pred' in net.params)

        #filename = (self.solver_param.snapshot_prefix +'_iter_{:d}'.format(self.solver.iter) + '.h5')
        filename = (self.solver_param.snapshot_prefix +'_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
                
        filename = os.path.join(self.output_dir, filename)
        
        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()
            
            means = self.bbox_means.ravel()
            stds = self.bbox_stds.ravel()
            
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data * stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                (net.params['bbox_pred'][1].data * stds + means)
        
        # we save as HDF5 format to reduce size
        #net.save_to_hdf5(str(filename))
        #net.save_hdf5(str(filename))
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)
        
        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            """
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % self.cfg.TRAIN_DEFAULT_SNAPSHOT_ITERS == 0: 
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())
            #"""
            
            #"""
            try:
                #print 'solver.iter: {}'.format(self.solver.iter)
                timer.tic()
                self.solver.step(1)
                timer.toc()
                if self.solver.iter % (10 * self.solver_param.display) == 0:
                    print 'speed: {:.3f}s / iter'.format(timer.average_time)

                if self.solver.iter % self.cfg.TRAIN_DEFAULT_SNAPSHOT_ITERS == 0: 
                    last_snapshot_iter = self.solver.iter
                    model_paths.append(self.snapshot())
            except Exception as e:
                print ("[ERROR] SolverWrapper.train_model > exception found: {}".format(e))
            #"""

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths
    
    
