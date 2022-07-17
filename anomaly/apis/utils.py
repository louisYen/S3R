
import os
from termcolor import colored

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def color(text, txt_color='green', attrs=['bold']):
    return colored(text, txt_color, attrs=attrs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
