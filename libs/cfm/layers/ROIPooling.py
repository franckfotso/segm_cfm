# Project: segm_cfm
# Module: libs.cfm.layers.ROIPooling
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Licensed under MIT License

import caffe, yaml
import numpy as np

class ROIPooling(caffe.Layer):
    """
        ROI Pooling layer
    """
    
    def setup(self, bottom, top):
        """
            ROIPooling layer > setup the layer
        """
        # parse the layer parameter string, which must be valid YAML
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)
        except Exception as f:
            raise Exception('[ERROR] exception found: {}'.format(f))   
               
        self.pooled_h = layer_params['pooled_h']
        self.pooled_w = layer_params['pooled_w']
        self.spatial_scale = layer_params['spatial_scale']
        
        # top 1: conv_rois
        num_rois = bottom[0].data.shape[0]
        channels = bottom[1].data.shape[1]
        top[0].reshape(num_rois, channels, self.pooled_h, self.pooled_w)
    
    def reshape(self, bottom, top):
        pass
    
    
    def forward(self, bottom, top):
        """
            ROIPooling layer > forward data cross the net
        """
        # ROIs built from input layer
        rois = bottom[0].data
        num_rois = rois.shape[0]
        # feature map based a full image
        im_conv = bottom[1].data
        channels = im_conv.shape[1]
        
        rois_conv = []
        # rois_conv_blob shape `(1, num_rois, channels, pool_size, pool_size)`
        out_shape = (num_rois, channels, self.pooled_h, self.pooled_w)
        #rois_conv_blob = np.zeros((1, num_rois, channels, self.pooled_w, self.pooled_h))
        
        for roi_id in xrange(num_rois):
            # project bbox to feature map scale
            bbox = (rois[roi_id, 1:5]*self.spatial_scale).astype(np.int32)
            h = bbox[3]-bbox[1]
            w = bbox[2]-bbox[0]
            """
            print 'bbox : {}'.format(bbox)
            print 'h,w : {},{}'.format(h,w)
            print 'im_conv.shape : {}'.format(im_conv.shape)
            print '----------------------------'
            """
            col_length = w / float(self.pooled_w)
            row_length = h / float(self.pooled_h)
                        
            num_pool_regions = min(self.pooled_w,self.pooled_h)
            
            for jy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    x1 = (bbox[0] + ix * col_length).astype(np.int32)
                    x2 = (x1 + col_length).astype(np.int32)
                    y1 = (bbox[1] + jy * row_length).astype(np.int32)
                    y2 = (y1 + row_length).astype(np.int32)
                    
                    """    
                    x2 = x1 + max(1,x2-x1)
                    y2 = y1 + max(1,y2-y1)
                    #"""
                    
                    #"""
                    if y2 < im_conv.shape[2]:
                        y2 = y1 + max(1,y2-y1)
                    else:
                        y2 = im_conv.shape[2]
                        y1 = y2 - 1
                        
                    if x2 < im_conv.shape[3]:
                        x2 = x1 + max(1,x2-x1)
                    else:
                        x2 = im_conv.shape[3]
                        x1 = x2 - 1
                    #"""                   
                    
                    top_shape = [im_conv.shape[0], im_conv.shape[1],
                                 y2 - y1, x2 - x1]

                    conv_crop = im_conv[:, :, y1:y2, x1:x2]
                        
                    try:
                        conv_crop = np.reshape(conv_crop, top_shape)
                        pooled_val = np.max(conv_crop, axis=(2, 3))
                        rois_conv.append(pooled_val)
                    except Exception as e:
                        print 'im_conv.shape : {}'.format(im_conv.shape)
                        print 'conv_crop.shape : {}'.format(conv_crop.shape)
                        print 'top_shape : {}'.format(top_shape)
                        print 'y1:y2={}:{}, x1:x2={}:{}'.format(y1,y2,x1,x2)
                        print 'col_length : {}'.format(col_length)
                        print 'row_length : {}'.format(row_length)
                        raise Exception("[ERROR] exception found: {}".format(e))
                        
        
        rois_conv = np.array(rois_conv)
        #print 'rois_conv.shape : {}'.format(rois_conv.shape)
        
        #rois_conv_blob = np.concatenate((rois_conv_blob, rois_conv), axis=0)
        rois_conv_blob = np.reshape(np.array(rois_conv), out_shape)
        
        # conv_rois: rois -> conv
        top[0].reshape(*rois_conv_blob.shape)
        top[0].data[...] = rois_conv_blob
        
    def backward(self, top, propagate_down, bottom):
        pass
    
    
