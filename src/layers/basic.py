import torch.nn as nn
from torch.nn.utils import spectral_norm
from .__init__ import get_norm, get_activ, get_padding
from inspect import signature
import torch
from math import sqrt
import torch.nn.functional as F
from .normalization import InstanceNorm
import numpy as np

def is_affine(norm_func):
    func_sig = signature(norm_func)
    if 'affine' in func_sig.parameters:
        affine_value = str(func_sig.parameters['affine']).split('=')[1]
        if affine_value == 'True':
            return True
    return False


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, pad='none', act='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = get_activ(act)
        self.normalization = get_norm(norm)
        self.padding = get_padding(pad)
        self._build_layers(ic, oc, ks, s, d)

    def _build_layers(self, ic, oc, ks, s, d):
        _layers = []
        pad_size = int(np.ceil((ks - 1.0) / 2))
        if self.padding is not None:
            _layers = _layers + [self.padding(pad_size)]
        if self.spectral:
            _layers = _layers + [spectral_norm(nn.Conv2d(ic, oc, ks, s, 0, d))]
        elif not self.spectral:
            _layers = _layers + [nn.Conv2d(ic, oc, ks, s, 0, d)]
        if self.activation is not None:
            _layers = _layers + [self.activation()]
        if self.normalization is not None:
            if self.spectral and is_affine(self.normalization):
                _layers = _layers + [spectral_norm(self.normalization(oc))]
            _layers = _layers + [self.normalization(oc)]
        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv(x)
        return x

class GLUConvBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, pad='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = nn.GLU(dim=1)
        self.normalization = get_norm(norm)
        self.padding = get_padding(pad)
        self._build_layers(ic, oc, ks, s, d)

    def _build_layers(self, ic, oc, ks, s, d):
        _layers = []
        pad_size = int(np.ceil((ks - 1.0) / 2))
        if self.padding is not None:
            _layers = _layers + [self.padding(pad_size)]
        if self.spectral:
            _layers = _layers + [spectral_norm(nn.Conv2d(ic, oc * 2, ks, s, 0, d))]
        elif not self.spectral:
            _layers = _layers + [nn.Conv2d(ic, oc * 2, ks, s, 0, d)]
        _layers = _layers + [self.activation]
        if self.normalization is not None:
            if self.spectral and is_affine(self.normalization):
                _layers = _layers + [spectral_norm(self.normalization(oc))]
            _layers = _layers + [self.normalization(oc)]
        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        return self.conv(x)

class GLUUpsampleBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, pad='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = nn.GLU(dim=1)
        self.normalization = norm
        self.padding = pad
        self._build_layers(ic, oc, ks, s, d)

    def _build_layers(self, ic, oc, ks, s, d):
        _layers = [nn.Upsample(scale_factor=2, mode='nearest')]
        _layers = _layers + [GLUConvBlock(ic, oc, ks, s, d, self.padding, self.normalization, self.spectral)]
        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv(x)
        return x

class GLUResBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, pad='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        #self.activation = get_activ(act)
        self.normalization = get_norm(norm)
        self.padding = get_padding(pad)
        self._build_layers(ic, oc, ks, s, d, pad, norm)

    def _build_layers(self, ic, oc, ks, s, d, pad, norm):
        _layers = []
        f = max(ic, oc)
        for _ in range(2):
            _layers = _layers + [GLUConvBlock(f, f, ks, s, d, pad, norm, self.spectral)]
        if oc != ic:
            self._residual = GLUConvBlock(ic, oc, ks, s, d, pad, norm, self.spectral)
            if ic > oc:
                _layers = _layers + [GLUConvBlock(ic, oc, ks, s, d, pad, norm, self.spectral)]
            elif oc > ic:
                _layers = [GLUConvBlock(ic, oc, ks, s, d, pad, norm, self.spectral)] + _layers
        else:
            _layers = _layers + [GLUConvBlock(ic, oc, ks, s, d, pad, norm, self.spectral)]
            self._residual = None
        self._conv = nn.Sequential(*_layers)

    def forward(self, x):
        if self._residual:
            res = self._residual(x)
        else:
            res = x
        x = res + self._conv(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, pad='none', act='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = act
        self.normalization = norm
        self.padding = pad
        self._build_layers(ic, oc, ks, s, d)

    def _build_layers(self, ic, oc, ks, s, d):
        _layers = [nn.Upsample(scale_factor=2, mode='bilinear')]
        _layers = _layers + [ConvBlock(ic, oc, ks, s, d, self.padding, self.activation, self.normalization, spectral=self.spectral)]

        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(self, ic, oc, ks=4, s=2, d=1, pad='none', act='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = get_activ(act)
        self.normalization = get_norm(norm)
        self.padding = get_padding(pad)
        self._build_layers(ic, oc, ks, s, d)

    def _build_layers(self, ic, oc, ks, s, d):
        _layers = []
        if self.padding is not None:
            _layers = _layers + [self.padding(int(ks // 2) - 1)]
        if self.spectral:
            _layers = _layers + [spectral_norm(nn.ConvTranspose2d(ic, oc, ks, s, 0, d))]
        else:
            _layers = _layers + [nn.ConvTranspose2d(ic, oc, ks, s, 0, d)]
        if self.normalization is not None:
            if self.spectral and is_affine(self.normalization):
                _layers = _layers + [spectral_norm(self.normalization(oc))]
            _layers = _layers + [self.normalization(oc)]
        if self.activation is not None:
            _layers = _layers + [self.activation()]
        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, pad='none', act='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = get_activ(act)
        self.normalization = get_norm(norm)
        self.padding = get_padding(pad)
        self._build_layers(ic, oc, ks, s, d, pad, act, norm)

    def _build_layers(self, ic, oc, ks, s, d, pad, act, norm):
        _layers = []
        f = max(ic, oc)
        for _ in range(2):
            _layers = _layers + [ConvBlock(f, f, ks, s, d, pad, act, norm, self.spectral)]
        if oc != ic:
            self._residual = ConvBlock(ic, oc, ks, s, d, pad, act, norm, self.spectral)
            if ic > oc:
                _layers = _layers + [ConvBlock(ic, oc, ks, s, d, pad, act, norm, self.spectral)]
            elif oc > ic:
                _layers = [ConvBlock(ic, oc, ks, s, d, pad, act, norm, self.spectral)] + _layers
        else:
            _layers = _layers + [ConvBlock(ic, oc, ks, s, d, pad, act, norm, self.spectral)]
            self._residual = None
        self._conv = nn.Sequential(*_layers)

    def forward(self, x):
        if self._residual:
            res = self._residual(x)
        else:
            res = x
        x = res + self._conv(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size, bias=True, act='none', norm='none', spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = get_activ(act)
        self.normalization = get_norm(norm)
        self._build_layers(input_size, output_size, bias)

    def _build_layers(self, input_size, output_size, bias):
        _layers = []
        if self.spectral:
            _layers = _layers + [spectral_norm(nn.Linear(input_size, output_size, bias))]
        elif not self.spectral:
            _layers = _layers + [nn.Linear(input_size, output_size, bias)]
        if self.activation is not None:
            _layers = _layers + [self.activation()]
        if self.normalization is not None:
            if self.spectral and is_affine(self.normalization):
                _layers = _layers + [spectral_norm(self.normalization(output_size))]
            _layers = _layers + [self.normalization(output_size)]
        self.linear = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.linear(x)
        return x


class EqualisedLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True, init_gain=2 ** 0.5, weight_scaling=True, custom_lr=1, ):
        super().__init__()
        # 2015 He initialization
        param_std = init_gain * input_size ** (-0.5)
        if weight_scaling:
            param_std = 1 / custom_lr
            self.weight_multiplier = param_std * custom_lr
        else:
            param_std = param_std / custom_lr
            self.weight_multiplier = custom_lr
        self.weight = nn.Parameter(torch.randn(size=(output_size, input_size)) * param_std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(output_size,)))
            self.bias_multiplier = custom_lr
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        weight = self.weight * self.weight_multiplier
        if bias is not None:
            bias = bias * self.bias_multiplier
        return F.linear(x, weight, bias)

class EqualisedConvLayer(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, bias=True, init_gain=2 ** 0.5, weight_scaling=True, custom_lr=1):
        super().__init__()
        param_std = init_gain * (ic * ks ** 2) ** (-0.5)
        if weight_scaling:
            param_std = 1 / custom_lr
            self.weight_multiplier = param_std * custom_lr
        else:
            param_std = param_std / custom_lr
            self.weight_multiplier = custom_lr
        self.weight = torch.nn.Parameter(torch.randn(size=(oc, ic, ks, ks)) * param_std)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(size=(oc,)))
            self.bias_multiplier = custom_lr
        else:
            self.bias = None

        self.stride = s
        self.dilation = d

    def forward(self, x, add_bias=True):
        # add_bias=True, indicates that the layer should automatically add the channel bias
        # add_bias=False, if there is a bias Parameter will not add it immediately after convolving
        weight = self.weight * self.weight_multiplier
        if add_bias:
            bias = self.bias
            if bias is not None:
                bias = bias * self.bias_multiplier
            return F.conv2d(x, weight, bias, stride=self.stride, dilation=self.dilation)
        else:
            return F.conv2d(x, weight, None, stride=self.stride, dilation=self.dilation)

    def get_bias(self):
        return self.bias.view(1, -1, 1, 1) * self.bias_multiplier

class Attention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, att_dim, spectral=False):
        super().__init__()
        self.spectral = spectral

        qc, qh, qw = q_dim
        kc, kh, kw = k_dim
        vc, vh, vw = v_dim

        self.sqrt = sqrt(att_dim)

        self._wq = nn.Linear(qc, att_dim)
        self._wk = nn.Linear(kc, att_dim)
        self._wv = nn.Linear(vc, att_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V):
        Q = Q.view(-1, Q.size(2) * Q.size(3), Q.size(1))
        K = K.view(-1, K.size(2) * K.size(3), K.size(1))
        V = V.view(-1, V.size(2) * V.size(3), V.size(1))

        proj_q = self._wq(Q)
        proj_k = self._wk(K)
        proj_v = self._wv(V)
        soft_score = proj_q @ proj_k.transpose(dim0=2, dim1=1)
        soft_score = soft_score / self.sqrt
        soft_score = self.softmax(soft_score)
        attn_score = soft_score @ proj_v
        print(attn_score.size())
        return attn_score


class ConvolutionalAttention(nn.Module):
    def __init__(self, in_channels, key_channels, val_channels, spectral=False):
        super().__init__()
        self._wq = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self._wk = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self._wv = nn.Conv2d(in_channels, val_channels, kernel_size=1)
        self.sqrt = sqrt(key_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        proj_q = self._wq(x)
        proj_k = self._wk(x)
        proj_v = self._wv(x)

        soft_score = proj_q @ proj_k.transpose(dim0=2, dim1=3)
        soft_score = soft_score / self.sqrt
        soft_score = self.softmax(soft_score)
        attn_score = soft_score @ proj_v
        print(attn_score.size())
        return attn_score


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, att_dim, attn_heads, out_dim, spectral=False):
        super().__init__()
        self.spectral = spectral
        self.heads = nn.ModuleList([Attention(q_dim, k_dim, v_dim, att_dim, spectral) for _ in range(attn_heads)])
        self.oc, self.oh, self.ow = out_dim
        self._wo = nn.Linear(att_dim * attn_heads, att_dim * attn_heads)

    def forward(self, Q, K, V):
        results = [head(Q, K, V) for head in self.heads]
        results = torch.cat(results, dim=2)
        results_proj = self._wo(results)
        return results_proj


class MultiHeadConvolutionalAttention(nn.Module):
    def __init__(self, in_channels, key_channels, val_channels, attn_heads, out_channels, spectral=False):
        super().__init__()
        self.spectral = spectral
        self.heads = nn.ModuleList([ConvolutionalAttention(in_channels, key_channels, val_channels, spectral) for _ in range(attn_heads)])
        self._wo = nn.Conv2d(in_channels=attn_heads * val_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        results = [head(x) for head in self.heads]
        results = torch.cat(results, dim=1)
        results_proj = self._wo(results)
        return results_proj