import os
import sys
import inspect
import os.path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

currentdir = osp.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = osp.dirname(currentdir)
sys.path.insert(0, parentdir)

# # Add lib to PYTHONPATH
# lib_path = osp.join(this_dir, 'lib')
# add_path(lib_path)
