import torch.nn as nn
from functools import partial
from .normalization import *


def get_padding(type='zeros'):
    if type == 'zeros':
        pad_layer = partial(nn.ZeroPad2d)
    elif type == 'reflect':
        pad_layer = partial(nn.ReflectionPad2d)
    elif type == 'replicate':
        pad_layer = partial(nn.ReplicationPad2d)
    elif type == 'none':
        pad_layer = None
    return pad_layer


def get_norm(type='instance'):
    if type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif type == 'instance':
        norm_layer = partial(InstanceNorm2d, affine=True)
    elif type == 'none':
        norm_layer = None
    elif type == 'layer':
        norm_layer = partial(nn.LayerNorm, elementwise_affine=True)
    else:
        norm_layer = partial(IdentityNorm)
    return norm_layer


def get_activ(type='relu'):
    if type == 'relu':
        activ_layer = partial(nn.ReLU, inplace=True)
    elif type == 'leaky_relu':
        activ_layer = partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif type == 'elu':
        activ_layer = partial(nn.ELU, alpha=0.2, inplace=True)
    elif type == 'tanh':
        activ_layer = partial(nn.Tanh)
    elif type == 'gelu':
        activ_layer = partial(nn.GELU)
    elif type == 'glu':
        activ_layer = partial(nn.GLU)
    elif type == 'sigmoid':
        activ_layer = partial(nn.Sigmoid)
    elif type == 'none':
        activ_layer = None
    return activ_layer
