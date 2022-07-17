
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, Sigmoid

class SigmoidMAELoss(nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)

