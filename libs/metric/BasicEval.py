# Project: segm_cfm
# Module: libs.metric.PascalEval
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Licensed under MIT License

import numpy as np

class BasicEval(object):
    
    def __init__(self, dataset, task, cfg):
        self._dataset = dataset
        self._task = task
        self._cfg = cfg     

    def get_cfg(self):
        return self._cfg


    def set_cfg(self, value):
        self._cfg = value


    def del_cfg(self):
        del self._cfg


    def get_dataset(self):
        return self._dataset


    def get_task(self):
        return self._task


    def set_dataset(self, value):
        self._dataset = value


    def set_task(self, value):
        self._task = value


    def del_dataset(self):
        del self._dataset


    def del_task(self):
        del self._task
        
        
    def voc_ap(self, rec, prec, use_07_metric=False):
        """
        Compute VOC AP given precision and recall. If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        Args:
            rec: recall
            prec: precision
            use_07_metric:
        Returns:
            ap: average precision
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap += p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))
    
            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]
    
            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    
    def coco_ap(self):
        raise NotImplemented

    dataset = property(get_dataset, set_dataset, del_dataset, "dataset's docstring")
    task = property(get_task, set_task, del_task, "task's docstring")
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
    
    