# Project: segm_cfm
# Module: libs.cfm.layers.MaskPooling
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Licensed under MIT License

import caffe, yaml, cv2
import numpy as np

class MaskPooling(caffe.Layer):
    """
        Mask Pooling layer
    """
    
    def setup(self, bottom, top):
        """
            MaskPooling layer > setup the layer
        """ 
        conv_rois = bottom[1].data
        n_roi, c, h, w = conv_rois.shape
        #print 'MaskPooling, conv_rois.shape: {}'.format(conv_rois.shape)
        top[0].reshape(n_roi, c, h, w)
    
    def reshape(self, bottom, top):
        pass    
    
    def forward(self, bottom, top):
        """
            ROIPooling layer > forward data cross the net
        """
        # masks: shape `(1, num_masks, channels, pool_size, pool_size)`
        masks = bottom[0].data      
        # conv_rois: shape `(num_rois, channels, pool_size, pool_size)`
        conv_rois = bottom[1].data
        num_rois = conv_rois.shape[0]
        channels = conv_rois.shape[1]
        
        #print 'MaskPooling, masks.shape: {}'.format(masks.shape)
        #print 'MaskPooling, conv_rois.shape: {}'.format(conv_rois.shape)
        
        assert masks.shape[0] == conv_rois.shape[0], \
            '[ERROR] conv_rois & masks should have same size: {} != {}'\
            .format(conv_rois.shape[0], masks.shape[0], )
        """    
        assert masks.shape[0] == 1, \
            '[ERROR] MaskPooling: only a single image can be handled, found: {}'.format(masks.shape[0])
        """    
        masks_pooled = np.zeros(conv_rois.shape)
        for m_i in xrange(num_rois):
            #masks_pooled[m_i,0:channels,:,:] = masks[0,m_i,0,:,:]
            masks_pooled[m_i,0:channels,:,:] = masks[m_i,0,:,:]
               
        """ mask out conv_rois and get conv_masks: CONV_ROIS * MASK_POOLED"""
        masks_conv_blob = np.multiply(conv_rois, masks_pooled)
        
        # conv_masks: conv_rois*masks_pooled
        top[0].reshape(*masks_conv_blob.shape)
        top[0].data[...] = masks_conv_blob
    
    def backward(self, top, propagate_down, bottom):
        pass
    
    
