import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'externals','caffe-segm', 'python')
add_path(caffe_path)

# Add selective_search to PYTHONPATH
selective_search_path = osp.join(this_dir, '..', 'externals','selective_search_py')
add_path(selective_search_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'libs')
add_path(lib_path)
