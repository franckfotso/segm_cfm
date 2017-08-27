# Project: 
# Module: libs.datasets.Mask
# Written by: romyny
# Licensed under MIT License
# Date: 17 Jun 2017

from datasets.BasicInput import BasicInput

class Mask(BasicInput):
    
    def __init__(self,
                 filename, 
                 pathname,
                 im_RAW=None,
                 gt_data=None,
                 pr_data=None
                 ):
        self._gt_data = gt_data
        self._pr_data = pr_data
        
        super(Mask, self).__init__(filename, pathname, im_RAW)

    def get_pr_data(self):
        return self.__pr_data


    def set_pr_data(self, value):
        self.__pr_data = value


    def del_pr_data(self):
        del self.__pr_data


    def get_gt_data(self):
        return self._gt_data


    def set_gt_data(self, value):
        self._gt_data = value


    def del_gt_data(self):
        del self._gt_data

    gt_data = property(get_gt_data, set_gt_data, del_gt_data, "gt_data's docstring")
    pr_data = property(get_pr_data, set_pr_data, del_pr_data, "pr_data's docstring")

        
    