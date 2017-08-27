# Project: segm_cfm
# Module: libs.datasets.Image
# Written by: romyny
# Licensed under MIT License
# Date: 17 Jun 2017

from datasets.BasicInput import BasicInput

class Image(BasicInput):
    
    def __init__(self,
                 filename, 
                 pathname,
                 im_RAW=None,
                 gt_rois=None,
                 pr_rois=None,
                 flipped=False
                 ):
        self._gt_rois = gt_rois
        self._pr_rois = pr_rois
        self._flipped = flipped
        
        super(Image, self).__init__(filename, pathname, im_RAW)

    def get_flipped(self):
        return self.__flipped


    def set_flipped(self, value):
        self.__flipped = value


    def del_flipped(self):
        del self.__flipped


    def get_pr_rois(self):
        return self._pr_rois


    def set_pr_rois(self, value):
        self._pr_rois = value


    def del_pr_rois(self):
        del self._pr_rois


    def get_gt_rois(self):
        return self._gt_rois


    def set_gt_rois(self, value):
        self._gt_rois = value


    def del_gt_rois(self):
        del self._gt_rois

    gt_rois = property(get_gt_rois, set_gt_rois, del_gt_rois, "gt_rois's docstring")
    pr_rois = property(get_pr_rois, set_pr_rois, del_pr_rois, "pr_rois's docstring")
    flipped = property(get_flipped, set_flipped, del_flipped, "flipped's docstring")
        
    