
import math
import torch
import numpy as np

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from einops import rearrange

# >> relu-based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class enNormalModule(nn.Module):
    def __init__(
        self,
        dim: int = 2048, # input feature dimension
        shrink_thres: float = 0.0025, # hyperparameter for sparse relu
    ):
        super(enNormalModule, self).__init__()

        self.shrink_thres: float = shrink_thres

        self.query_embedding = nn.Linear(dim, dim // 4)
        self.cache_embedding = nn.Linear(dim, dim // 4)
        self.value_embedding = nn.Linear(dim, dim)


    def forward(
            self,
            query, # input video feature of shape (BN)TC, N=num_crops (default=10)
            cache, # memorized normal video feature of shape BSC, S=num_slots
        ):

        _, T, C = query.shape
        B = cache.shape[0]

        num_slots = cache.size(1)
        num_crops = query.shape[0] // B

        query = rearrange(query, '(b n) t c -> b (n t) c', b=B) # B(NT)C

        x_query = self.query_embedding(query) # BLD=B(NT)D, D=embedding dim
        x_cache = self.cache_embedding(cache) # BSD
        x_value = self.value_embedding(cache) # BSC

        affinity = x_query @ x_cache.transpose(1, 2) # BLS
        affinity = affinity.softmax(dim=-1)  # BLS

        # =======================================================
        # ReLU based shrinkage, hard shrinkage for positive value
        # -------------------------------------------------------
        if self.shrink_thres > 0:
            affinity = hard_shrink_relu(affinity, lambd = self.shrink_thres)
            # normalize
            affinity = F.normalize(affinity, p=1, dim=-1) # BLS

        out = affinity @ x_value # BLC
        out = rearrange(out, 'b (n t) c -> (b n) t c', t=T) # (BN)TC

        return out, affinity

class enNormal(nn.Module):
    def __init__(
        self,
        dim: int = 2048,
        shrink_thres: float = 0.0025,
        num_univ: int = 1001,
        modality: str = 'taskaware',
    ):
        super(enNormal, self).__init__()

        self.num_univ: int = num_univ
        self.modality: int = modality
        self.en_normal_module = enNormalModule(dim, shrink_thres=0.)

    def forward(
            self,
            video: Tensor, # input video of shape (BN)TC, N=num_crops (default=10)
            macro: Tensor, # memory feature of shape BSC, S=num_slots
        ):

        num_univ = self.num_univ
        modality = self.modality

        out, attn = self.en_normal_module(query=video, cache=macro)
        attn = attn.transpose(1, 2) # BST

        return out, attn
