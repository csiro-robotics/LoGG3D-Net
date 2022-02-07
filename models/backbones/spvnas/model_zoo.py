import os
import sys
import json
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

import torch
from torchpack import distributed as dist
from core.models.semantic_kitti.spvcnn import SPVCNN


__all__ = ['spvcnn']



def spvcnn(output_dim=16):

    model = SPVCNN(
        num_classes=output_dim,
        cr=0.64,
        pres=0.05,
        vres=0.05
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')


    return model

