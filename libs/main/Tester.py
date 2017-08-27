# Project: segm_cfm
# Module: libs.main.Tester
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Licensed under MIT License

from BasicWorker import BasicWorker
from TestWrapper import TestWrapper
from datasets.CFMGenerator import CFMGenerator

class Tester(BasicWorker):
    
    def __init__(self,
                 dataset,
                 model,
                 task):
        
        super(Tester, self).__init__(dataset,
                 model,
                 task)
        
    def run(self, gt_set, images, masks, backbone_net, 
            task, in_pr_dir, cache_im_dir, output_dir, kwargs=None):
                
        assert task in self.cfg.MAIN_DEFAULT_TASKS, \
            '[ERROR] unknown task name provided: {}'.format(self.task)
        
        if task == 'CFM':        
            data_gen = CFMGenerator(images=images, masks=masks,
                                    bbox_means=None,
                                    bbox_stds=None,
                                    num_cls=self.dataset.num_cls, cfm_t=None,
                                    in_gt_dir=None, in_pr_dir=in_pr_dir,
                                    cache_im_dir=cache_im_dir, cfg=self.cfg)
        elif self.task == 'MNC':
            raise NotImplemented
        
        elif self.task == 'FCIS':
            raise NotImplemented
        
        elif self.task == 'MRCNN':
            raise NotImplemented    
        
        
        tw = TestWrapper(self.model['proto'], 
                         self.model['weights'], task,
                         data_gen, output_dir, self.cfg)

        print 'Testing...'
        tw.test_model(gt_set, backbone_net, self.dataset, True, kwargs)
        print 'done testing'
            
    
    