import math
import torch
from torch import nn


def preserve_initialization_rng(dimension, count):
    for _ in range(count):
        nn.init.kaiming_uniform_(torch.empty(dimension, dimension), a=math.sqrt(5))
