import numpy as np
import os

class Config(object):
    def __init__(self, lr):
        self.lr = lr
        # self.lr_str = lr

        self.envrows, self.envcols = list(map(int, os.popen('stty size', 'r').read().split()))

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')


class Obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
               setattr(self, k, [Obj(x) if isinstance(x, dict) else x for x in v])
            else:
               setattr(self, k, Obj(v) if isinstance(v, dict) else v)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

