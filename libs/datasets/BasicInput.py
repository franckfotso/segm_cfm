# Project: 
# Module: libs.datasets.BasicInput
# Written by: Romuald FOTSO
# Licensed under MIT License
# Date: 17 Jun 2017

class BasicInput(object):
    
    def __init__(self, 
                 filename, 
                 pathname,
                 im_RAW):
        self._filename = filename
        self._pathname = pathname
        self._im_RAW = im_RAW
        
    def get_filename(self):
        return self._filename


    def get_pathname(self):
        return self._pathname


    def get_im_raw(self):
        return self._im_RAW


    def set_filename(self, value):
        self._filename = value


    def set_pathname(self, value):
        self._pathname = value


    def set_im_raw(self, value):
        self._im_RAW = value


    def del_filename(self):
        del self._filename


    def del_pathname(self):
        del self._pathname


    def del_im_raw(self):
        del self._im_RAW

    filename = property(get_filename, set_filename, del_filename, "filename's docstring")
    pathname = property(get_pathname, set_pathname, del_pathname, "pathname's docstring")
    im_RAW = property(get_im_raw, set_im_raw, del_im_raw, "im_RAW's docstring")
