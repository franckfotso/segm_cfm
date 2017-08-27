# Project: 
# Module: libs.main.BasicWorker
# Written by: Romuald FOTSO
# Licensed under MIT License
# Date: 17 Jun 2017

class BasicWorker(object):
    
    def __init__(self,                 
                 dataset,
                 model,
                 cfg
                 ):
        self._dataset = dataset
        self._model = model
        self._cfg = cfg

    def get_dataset(self):
        return self._dataset


    def get_model(self):
        return self._model


    def get_cfg(self):
        return self._cfg


    def set_dataset(self, value):
        self._dataset = value


    def set_model(self, value):
        self._model = value


    def set_cfg(self, value):
        self._cfg = value


    def del_dataset(self):
        del self._dataset


    def del_model(self):
        del self._model


    def del_cfg(self):
        del self._cfg

    dataset = property(get_dataset, set_dataset, del_dataset, "dataset's docstring")
    model = property(get_model, set_model, del_model, "model's docstring")
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
        
    
        