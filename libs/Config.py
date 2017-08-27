# Project: segm_cfm
# Module: libs.Config
# Written by: Romuald FOTSO
# Licensed under MIT License
# Date: 18 Jun 2017

import ConfigParser as cp
import importlib
from easydict import EasyDict as edict

class Config(object):
    
    def __init__(self, 
                 config_pn="../config/config.ini",
                 extras=None):
        self._parser = cp.RawConfigParser()
        self._parser.read(config_pn)
        
        self._extras = extras
        self._cfg = edict()
        
        self.load_config()

    def get_parser(self):
        return self._parser


    def set_parser(self, value):
        self._parser = value


    def del_parser(self):
        del self._parser
        

    def get_cfg(self):
        return self._cfg


    def set_cfg(self, value):
        self._cfg = value


    def del_cfg(self):
        del self._cfg


    def get_extras(self):
        return self._extras


    def set_extras(self, value):
        self._extras = value


    def del_extras(self):
        del self._extras

    parser = property(get_parser, set_parser, del_parser, "parser's docstring")
    extras = property(get_extras, set_extras, del_extras, "extras's docstring")
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
    
    ''' 
        load_config: load config parameters from config.ini
    '''        
    def load_config(self):
        try:
            ''' 
                MAIN.DEFAULT
            '''
            self.cfg.MAIN_DEFAULT_GPU_ID = self.load_param("MAIN.DEFAULT", "GPU_ID", "int")
            self.cfg.MAIN_DEFAULT_RNG_SEED = self.load_param("MAIN.DEFAULT", "RNG_SEED", "int")
            self.cfg.MAIN_DEFAULT_TASK = self.load_param("MAIN.DEFAULT", "TASK")
            self.cfg.MAIN_DEFAULT_USE_GPU_NMS = self.load_param("MAIN.DEFAULT", "USE_GPU_NMS", "bool")
            self.cfg.MAIN_DEFAULT_PIXEL_MEANS = self.load_param("MAIN.DEFAULT", "PIXEL_MEANS", "list", "float")
            self.cfg.MAIN_DEFAULT_EPS = self.load_param("MAIN.DEFAULT", "EPS", "float")
            self.cfg.MAIN_DEFAULT_MASK_SIZE = self.load_param("MAIN.DEFAULT", "MASK_SIZE", "int")
            self.cfg.MAIN_DEFAULT_MASK_POOL_SHAPE = self.load_param("MAIN.DEFAULT", "MASK_POOL_SHAPE", "list","int")
            self.cfg.MAIN_DEFAULT_BINARIZE_THRESH = self.load_param("MAIN.DEFAULT", "BINARIZE_THRESH","float")
            self.cfg.MAIN_DEFAULT_TASKS = self.load_param("MAIN.DEFAULT", "TASKS","list")
            self.cfg.MAIN_DEFAULT_DATASETS = self.load_param("MAIN.DEFAULT", "DATASETS","list")
            ''' 
                MAIN.DIR
            '''
            self.cfg.MAIN_DIR_ROOT = self.load_param("MAIN.DIR", "ROOT")
            self.cfg.MAIN_DIR_CACHE = self.load_param("MAIN.DIR", "CACHE")
            self.cfg.MAIN_DIR_LOGS = self.load_param("MAIN.DIR", "LOGS")
            self.cfg.MAIN_DIR_OUTPUTS = self.load_param("MAIN.DIR", "OUTPUTS")
            '''
                SEGM.SS
            '''
            self.cfg.SEGM_SS_K = self.load_param("SEGM.SS", "K","int")
            self.cfg.SEGM_SS_FEATURE = self.load_param("SEGM.SS", "FEATURE","list")
            self.cfg.SEGM_SS_COLOR = self.load_param("SEGM.SS", "COLOR","list")
            self.cfg.SEGM_SS_DEPTH = self.load_param("SEGM.SS", "DEPTH","int")
            self.cfg.SEGM_SS_ALPHA = self.load_param("SEGM.SS", "ALPHA","float")
            '''
                SEGM.MCG
            '''
            
            '''
                SEGM.COB
            '''
            
            ''' 
                PASCAL.DATASET.DEFAULT
            '''
            self.cfg.PASCAL_DATASET_DEFAULT_EXT = self.load_param("PASCAL.DATASET.DEFAULT", "EXT", "list")
            ''' 
                PASCAL.DATASET.DIR
            '''
            self.cfg.PASCAL_DATASET_DIR_MAIN_SET = self.load_param("PASCAL.DATASET.DIR", "MAIN_SET")
            self.cfg.PASCAL_DATASET_DIR_SEGM_SET = self.load_param("PASCAL.DATASET.DIR", "SEGM_SET")
            self.cfg.PASCAL_DATASET_DIR_IMAGE = self.load_param("PASCAL.DATASET.DIR", "IMAGE")
            self.cfg.PASCAL_DATASET_DIR_ANNOTATION = self.load_param("PASCAL.DATASET.DIR", "ANNOTATION")
            self.cfg.PASCAL_DATASET_DIR_SEGM_CLS = self.load_param("PASCAL.DATASET.DIR", "SEGM_CLS")
            self.cfg.PASCAL_DATASET_DIR_SEGM_OBJ = self.load_param("PASCAL.DATASET.DIR", "SEGM_OBJ")
            self.cfg.PASCAL_DATASET_DIR_SEGM_SS = self.load_param("PASCAL.DATASET.DIR", "SEGM_SS")
            ''' 
                PASCAL.DATASET.FILE
            '''
            self.cfg.PASCAL_DATASET_FILE_VAL = self.load_param("PASCAL.DATASET.FILE", "VAL")
            self.cfg.PASCAL_DATASET_FILE_TEST = self.load_param("PASCAL.DATASET.FILE", "TEST")
            self.cfg.PASCAL_DATASET_FILE_TRAIN = self.load_param("PASCAL.DATASET.FILE", "TRAIN")
            self.cfg.PASCAL_DATASET_FILE_TRAINVAL = self.load_param("PASCAL.DATASET.FILE", "TRAINVAL")           
            self.cfg.PASCAL_DATASET_FILE_CLS = self.load_param("PASCAL.DATASET.FILE", "CLS")
            self.cfg.PASCAL_DATASET_FILE_SEGM = self.load_param("PASCAL.DATASET.FILE", "SEGM")
            
            ''' 
                TRAIN.DEFAULT
            '''
            self.cfg.TRAIN_DEFAULT_DEBUG = self.load_param("TRAIN.DEFAULT", "DEBUG", "bool")
            self.cfg.TRAIN_DEFAULT_USE_FLIPPED = self.load_param("TRAIN.DEFAULT", "USE_FLIPPED", "bool")
            self.cfg.TRAIN_DEFAULT_SNAPSHOT_ITERS = self.load_param("TRAIN.DEFAULT", "SNAPSHOT_ITERS", "int")
            self.cfg.TRAIN_DEFAULT_SEGM_METHOD = self.load_param("TRAIN.DEFAULT", "SEGM_METHOD")
            self.cfg.TRAIN_DEFAULT_SCALES = self.load_param("TRAIN.DEFAULT", "SCALES","list","int")
            self.cfg.TRAIN_DEFAULT_MAX_SIZE = self.load_param("TRAIN.DEFAULT", "MAX_SIZE","int")
            self.cfg.TRAIN_DEFAULT_ASPECT_GROUPING \
                = self.load_param("TRAIN.DEFAULT", "ASPECT_GROUPING","bool")
            self.cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_TARGETS \
                = self.load_param("TRAIN.DEFAULT", "BBOX_NORMALIZE_TARGETS","bool")
            self.cfg.TRAIN_DEFAULT_BBOX_INSIDE_WEIGHTS \
                = self.load_param("TRAIN.DEFAULT", "BBOX_INSIDE_WEIGHTS","list","float")
            self.cfg.TRAIN_DEFAULT_BBOX_REG = self.load_param("TRAIN.DEFAULT", "BBOX_REG","bool")
            self.cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_TARGETS_PRECOMPUTED \
                = self.load_param("TRAIN.DEFAULT", "BBOX_NORMALIZE_TARGETS_PRECOMPUTED","bool")
            self.cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_MEANS \
                = self.load_param("TRAIN.DEFAULT", "BBOX_NORMALIZE_MEANS","list","float")
            self.cfg.TRAIN_DEFAULT_BBOX_NORMALIZE_STDS \
                = self.load_param("TRAIN.DEFAULT", "BBOX_NORMALIZE_STDS","list","float")
            self.cfg.TRAIN_DEFAULT_BBOX_THRESH = self.load_param("TRAIN.DEFAULT", "BBOX_THRESH","float")
            self.cfg.TRAIN_DEFAULT_ROI_FG_THRESH = self.load_param("TRAIN.DEFAULT", "ROI_FG_THRESH","float")            
            self.cfg.TRAIN_DEFAULT_ROI_BG_THRESH_HI = self.load_param("TRAIN.DEFAULT", "ROI_BG_THRESH_HI","float")
            self.cfg.TRAIN_DEFAULT_ROI_BG_THRESH_LO = self.load_param("TRAIN.DEFAULT", "ROI_BG_THRESH_LO","float")    
            self.cfg.TRAIN_DEFAULT_SEG_FG_THRESH = self.load_param("TRAIN.DEFAULT", "SEG_FG_THRESH","float")            
            self.cfg.TRAIN_DEFAULT_MASk_TARGETS_POOLED = self.load_param("TRAIN.DEFAULT", "MASk_TARGETS_POOLED","bool")
            ''' 
                TRAIN.BATCH
            '''
            self.cfg.TRAIN_BATCH_NUM_IMG = self.load_param("TRAIN.BATCH", "NUM_IMG", "int")
            self.cfg.TRAIN_BATCH_NUM_ROI = self.load_param("TRAIN.BATCH", "NUM_ROI", "int")
            self.cfg.TRAIN_BATCH_NUM_SEGM = self.load_param("TRAIN.BATCH", "NUM_SEGM", "int")
            self.cfg.TRAIN_BATCH_FRACTION_SAMPLE = self.load_param("TRAIN.BATCH", "FRACTION_SAMPLE", "list", "float")
            self.cfg.TRAIN_BATCH_THRESH_LO_SAMPLE = self.load_param("TRAIN.BATCH", "THRESH_LO_SAMPLE", "list", "float")
            self.cfg.TRAIN_BATCH_THRESH_HI_SAMPLE = self.load_param("TRAIN.BATCH", "THRESH_HI_SAMPLE", "list", "float")
            
            ''' 
                TRAIN.LAYER
            '''
            
            ''' 
                TEST.DEFAULT
            '''
            self.cfg.TEST_DEFAULT_DEBUG = self.load_param("TEST.DEFAULT", "DEBUG", "bool")
            self.cfg.TEST_DEFAULT_USE_FLIPPED = self.load_param("TEST.DEFAULT", "USE_FLIPPED", "bool")
            self.cfg.TEST_DEFAULT_SEGM_METHOD = self.load_param("TEST.DEFAULT", "SEGM_METHOD")           
            self.cfg.TEST_DEFAULT_SCALES = self.load_param("TEST.DEFAULT", "SCALES","list","int")
            self.cfg.TEST_DEFAULT_ROI_MIN_SIZE = self.load_param("TEST.DEFAULT", "ROI_MIN_SIZE","int")
            self.cfg.TEST_DEFAULT_TOP_K_PROPOSALS = self.load_param("TEST.DEFAULT", "TOP_K_PROPOSALS","int")
            self.cfg.TEST_DEFAULT_GROUP_SCALE = self.load_param("TEST.DEFAULT", "GROUP_SCALE","int")
            self.cfg.TEST_DEFAULT_MAX_ROIS_GPU = self.load_param("TEST.DEFAULT", "MAX_ROIS_GPU", "list","int")
            self.cfg.TEST_DEFAULT_NMS = self.load_param("TEST.DEFAULT", "NMS", "float")
            self.cfg.TEST_DEFAULT_CONF_THRESH = self.load_param("TEST.DEFAULT", "CONF_THRESH", "float")
            self.cfg.TEST_DEFAULT_USE_GPU_MASK_MERGE = \
                self.load_param("TEST.DEFAULT", "USE_GPU_MASK_MERGE", "bool")
            self.cfg.TEST_DEFAULT_MASK_MERGE_IOU_THRESH = \
                self.load_param("TEST.DEFAULT", "MASK_MERGE_IOU_THRESH", "float")
            self.cfg.TEST_DEFAULT_MASK_MERGE_NMS_THRESH = \
                self.load_param("TEST.DEFAULT", "MASK_MERGE_NMS_THRESH", "float")
            
            
            ''' 
                EVAL.DEFAULT
            '''
            self.cfg.EVAL_DEFAULT_METRIC = self.load_param("EVAL.DEFAULT", "METRIC")
            ''' 
                EVAL.ROI
            '''
            
            ''' 
                EVAL.SEGM
            '''
            
        except Exception as e:
            print ("[Error] loading config: {}".format(str(e)))
    
    ''' 
        load_param: load casted parameters
    '''
    def load_param(self, bloc, param, proto="str", sub_proto="str"):
        
        if proto == "list":
            cls = None
            try:
                module = importlib.import_module('__builtin__')
                cls = getattr(module, sub_proto)
                
            except AttributeError:
                module, sub_proto = sub_proto.rsplit(".", 1)
                module = importlib.import_module(module)
                cls = getattr(module, sub_proto)    
                
            except Exception as e:
                print ("[Error]load_param: {}".format(str(e)))
                
            assert cls != None, "[Error] unable to load parameters: unknown type"
            
            vals = self.parser.get(bloc, param).split(",")
            vals = [ cls(v) for v in vals]
            return vals
        
        else:        
            cls = None       
            try:
                if proto == 'bool':
                    val = self.parser.get(bloc, param)
                    return val.lower() in ['true','yes','1']
                    
                module = importlib.import_module('__builtin__')
                cls = getattr(module, proto)
                
            except AttributeError:
                module, proto = proto.rsplit(".", 1)
                module = importlib.import_module(module)
                cls = getattr(module, proto)    
                
            except Exception as e:
                print ("[Error]load_param: {}".format(str(e)))
                
            assert cls != None, "[Error] unable to load parameters: unknown type"
            
            return cls(self.parser.get(bloc, param))
            
            
        
            