# Project: segm_cfm
# Module: libs.cfm.layers.InputData
# Written by: Romuald FOTSO
# Licensed under MIT License
# Date: 1 Jul 2017

import caffe, yaml

from Config import Config

import numpy as np


class InputData(caffe.Layer):
    """
        segm_cfm data layer used for training.
    """
    
    def setup(self, bottom, top):
        """
            InputData layer > setup the layer
        """
        #print('InputData > setup: start')
        _C = Config(config_pn="config/config.ini")
        self.cfg = _C.cfg
    
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)        
        self.num_cls = layer_params['num_cls']
        self.cfm_t = layer_params['cfm_t']        
        self.name_to_top_map = {}
        mask_in_h, mask_in_w = self.cfg.MAIN_DEFAULT_MASK_POOL_SHAPE
                
        if self.cfm_t == 'T1':          
            # top 1: data
            print('InputData > setup, cfm_t: {}'.format(self.cfm_t))
            top[0].reshape(self.cfg.TRAIN_BATCH_NUM_IMG, 3,
                max(self.cfg.TRAIN_DEFAULT_SCALES), 
                self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            self.name_to_top_map['data'] = 0            
            # top 2: rois
            top[1].reshape(1, 5)
            self.name_to_top_map['rois'] = 1            
            # top 3: masks
            top[2].reshape(1, 1, mask_in_h, mask_in_w)
            self.name_to_top_map['masks'] = 2
            # top 4: bbox_labels
            top[3].reshape(1, 1)
            self.name_to_top_map['bbox_labels'] = 3
            # top 5: mask_labels
            top[4].reshape(1, 1)
            self.name_to_top_map['mask_labels'] = 4
            
        elif self.cfm_t == 'T2':
            # top 1: data
            top[0].reshape(self.cfg.TRAIN_BATCH_NUM_IMG, 3,
                max(self.cfg.TRAIN_DEFAULT_SCALES), 
                self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            self.name_to_top_map['data'] = 0
            # top 2: rois
            top[1].reshape(1, 5)
            self.name_to_top_map['rois'] = 1  
            # top 3: masks
            top[2].reshape(1, 1, mask_in_h, mask_in_w)
            self.name_to_top_map['masks'] = 2
            # top 4: bbox_labels
            top[3].reshape(1, 1)
            self.name_to_top_map['bbox_labels'] = 3
            # top 5: mask_labels
            top[4].reshape(1, 1)
            self.name_to_top_map['mask_labels'] = 4
            # top 6: bbox_targets
            top[5].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_targets'] = 5
            # input 7: bbox_inside_weights
            top[6].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_inside_weights'] = 6
            # top 8: bbox_outside_weights
            top[7].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_outside_weights'] = 7
            
        elif self.cfm_t == 'T3':
            # top 1: data
            top[0].reshape(self.cfg.TRAIN_BATCH_NUM_IMG, 3,
                max(self.cfg.TRAIN_DEFAULT_SCALES), 
                self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            self.name_to_top_map['data'] = 0
            # top 2: rois
            top[1].reshape(1, 5)
            self.name_to_top_map['rois'] = 1  
            # top 3: masks
            top[2].reshape(1, 1, mask_in_h, mask_in_w)
            self.name_to_top_map['masks'] = 2
            # top 4: bbox_labels
            top[3].reshape(1, 1)
            self.name_to_top_map['bbox_labels'] = 3
            # top 5: mask_labels
            top[4].reshape(1, 1)
            self.name_to_top_map['mask_labels'] = 4
            # top 6: bbox_targets
            top[5].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_targets'] = 5
            # top 7: mask_targets
            top[6].reshape(1, self.cfg.MAIN_DEFAULT_MASK_SIZE*self.cfg.MAIN_DEFAULT_MASK_SIZE)
            self.name_to_top_map['mask_targets'] = 6
            # top 8: bbox_inside_weights
            top[7].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_inside_weights'] = 7
            # top 9: bbox_outside_weights
            top[8].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_outside_weights'] = 8
            # top 10: mask_weights
            #top[9].reshape(1, self.cfg.MAIN_DEFAULT_MASK_SIZE*self.cfg.MAIN_DEFAULT_MASK_SIZE)
            #self.name_to_top_map['mask_weights'] = 9
            
        elif self.cfm_t == 'T4':
            # top 1: data
            top[0].reshape(self.cfg.TRAIN_BATCH_NUM_IMG, 3,
                max(self.cfg.TRAIN_DEFAULT_SCALES), 
                self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            self.name_to_top_map['data'] = 0
            # top 2: rois
            top[1].reshape(1, 5)
            self.name_to_top_map['rois'] = 1  
            # top 3: masks
            top[2].reshape(1, 1, mask_in_h, mask_in_w)
            self.name_to_top_map['masks'] = 2
            # top 4: bbox_labels
            top[3].reshape(1, 1)
            self.name_to_top_map['bbox_labels'] = 3
            # top 5: mask_labels
            top[4].reshape(1, 1)
            self.name_to_top_map['mask_labels'] = 4
            # top 6: bbox_targets
            top[5].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_targets'] = 5
            # top 7: mask_targets
            top[6].reshape(1, self.cfg.MAIN_DEFAULT_MASK_SIZE*self.cfg.MAIN_DEFAULT_MASK_SIZE)
            self.name_to_top_map['mask_targets'] = 6
            # top 8: bbox_inside_weights
            top[7].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_inside_weights'] = 7
            # top 9: bbox_outside_weights
            top[8].reshape(1, self.num_cls*4)
            self.name_to_top_map['bbox_outside_weights'] = 8
            # top 10: mask_weights
            top[9].reshape(1, self.cfg.MAIN_DEFAULT_MASK_SIZE*self.cfg.MAIN_DEFAULT_MASK_SIZE)
            self.name_to_top_map['mask_weights'] = 9
            
        else:
            raise('[ERROR] unknown CFM type')
        
        print 'InputDataLayer: name_to_top:', self.name_to_top_map
        assert len(top) == len(self.name_to_top_map)        
        #print('InputData > setup: done')
    
    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        """
            InputData layer > forward data cross the net
        """
        blobs = self.data_gen.get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            if blob_name in self.name_to_top_map.keys():
                top_ind = self.name_to_top_map[blob_name]
                #print 'blob_name.shape: {}'.format(blob.shape)
                # Reshape net's input blobs
                top[top_ind].reshape(*(blob.shape))
                # Copy data into net's input blobs
                top[top_ind].data[...] = blob.astype(np.float32, copy=False)
    
    def backward(self, top, propagate_down, bottom):
        """
            InputData layer > backward (This layer does not propagate gradients)
        """
        pass
    
    def set_data_generator(self, data_gen):
        data_gen.num_cls = self.num_cls
        data_gen.cfm_t = self.cfm_t
        self.data_gen = data_gen
        self.cfg = data_gen.cfg
        
        
        
