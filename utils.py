import math
import numpy as np
import torch

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2*var) - 0.5 * math.log(2*math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
        
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    pass