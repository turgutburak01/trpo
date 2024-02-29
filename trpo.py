import numpy as np

import torch
from torch.autograd import Variable
from utils import *


def trpo_step(model, get_loss, get_kl, max_kl, damping):
    pass