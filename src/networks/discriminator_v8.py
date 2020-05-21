import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.basic import *
from ..layers.normalization import InstanceNorm2d
from functools import partial


class ViewLayer(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        out = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return out


class SpectralDecoupledConv(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, bias=True, use_pad=True):
        super().__init__()
        self.stride = s
        self.dilation = d
        if use_pad:
            self.pad_size = int(np.ceil((ks - 1.0) / 2))
        else:
            self.pad_size = 0
        self._conv = spectral_norm(nn.Conv2d(ic, oc, ks, s, padding=use_pad, dilation=d, bias=bias))

    def forward(self, x, add_bias=True):
        #if add_bias:
        #    bias = self._conv.bias
        #    if bias is not None:
        #        return F.conv2d(x, self._conv.weight_orig, bias, stride=self.stride, dilation=self.dilation, padding=self.pad_size)
        #else:
        #    # return self._conv(x)
        #    return F.conv2d(x, self._conv.weight_orig, None, stride=self.stride, dilation=self.dilation, padding=self.pad_size)
        return self._conv(x)

class SpectralLinear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self._linear = spectral_norm(nn.Linear(dim_in, dim_out, bias=bias))

    def forward(self, x):
        out = self._linear(x)
        return out


class SpectralWrapperNorm(nn.Module):
    def __init__(self, constructor, num_features):
        super().__init__()
        self._norm = spectral_norm(constructor(num_features))

    def forward(self, x):
        return self._norm(x)


class fRGB(nn.Module):
    def __init__(self, oc):
        super().__init__()
        self._transform = nn.Sequential(*[
            SpectralDecoupledConv(3, oc, 1),
        ])

    def forward(self, x):
        return self._transform(x)


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.real_label = cfg.real_label - cfg.smooth
        self.fake_label = cfg.fake_label
        self.generator_label = cfg.real_label
        self.loss = nn.BCEWithLogitsLoss()
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.code_tensor = None
        self.generator_label_tensor = None

    def get_target_tensor(self, input, target_is_real, for_discriminator):
        if for_discriminator:
            if target_is_real:
                if self.real_label_tensor is None:
                    self.real_label_tensor = torch.FloatTensor(1).fill_(self.real_label)
                    self.real_label_tensor.requires_grad_(False)
                return self.real_label_tensor.expand_as(input)
            else:
                if self.fake_label_tensor is None:
                    self.fake_label_tensor = torch.FloatTensor(1).fill_(self.fake_label)
                    self.fake_label_tensor.requires_grad_(False)
                return self.fake_label_tensor.expand_as(input)
        else:
            if self.generator_label_tensor is None:
                self.generator_label_tensor = torch.FloatTensor(1).fill_(self.generator_label)
                self.generator_label_tensor.requires_grad_(False)
            return self.generator_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def get_code_tensor(self, input, code):
        if self.code_tensor is None:
            self.code_tensor = torch.FloatTensor(1).fill_(1)
            self.code_tensor.requires_grad_(False)
        self.code_tensor = self.code_tensor * code
        self.code_tensor.requires_grad_(False)
        return self.code_tensor.expand_as(input)

    def __call__(self, input, target_is_real=True, for_discriminator=True):
        target_tensor = self.get_target_tensor(input, target_is_real, for_discriminator)
        loss = self.loss(input, target_tensor.to(input.device))
        return loss



class DownsampleLayer(nn.Module):
    def __init__(self, cfg, ic, oc, ks=3, f=2):
        super().__init__()
        self.cfg = cfg

        self._downsample = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=(f, f), stride=f),
            SpectralDecoupledConv(ic, oc, ks),
            nn.LeakyReLU(negative_slope=0.2),
        ])

    def forward(self, x):
        return self._downsample(x)


class FeatureConvolutionalDiscriminator(nn.Module):
    def __init__(self, cfg, scale):
        super().__init__()
        self.cfg = cfg
        self.input_dim = int(scale)
        self._norm = get_norm('batch')
        self._head_norm = partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
        self._build_feature_layers(cfg)
        self._build_u_head(cfg)
        self._build_c_head(cfg)

    def _build_feature_layers(self, cfg):
        _layers = [fRGB(cfg.ndf)]
        f = 1
        pf = 1

        while self.input_dim != 8:
            f = min(8, f * 2)
            _layers += [DownsampleLayer(cfg, cfg.ndf * pf, cfg.ndf * f), SpectralWrapperNorm(self._norm, cfg.ndf * f)]
            pf = f
            self.input_dim = int(self.input_dim / 2)

        _layers += [nn.AdaptiveAvgPool2d(output_size=(1, 1))]
        self._body = nn.Sequential(*_layers)

    def _build_u_head(self, cfg):
        _layers = []
        _layers += [
            SpectralLinear(8 * cfg.ndf, 8 * cfg.ndf),
            nn.LeakyReLU(negative_slope=0.2),
            SpectralWrapperNorm(self._head_norm, 8 * cfg.ndf),
            SpectralLinear(8 * cfg.ndf, 1)
        ]
        self._u_head = nn.Sequential(*_layers)

    def _build_c_head(self, cfg):
        _layers = []
        _layers += [
            SpectralLinear(16 * cfg.ndf, 16 * cfg.ndf),
            nn.LeakyReLU(negative_slope=0.2),
            SpectralWrapperNorm(self._head_norm, 16 * cfg.ndf),
            SpectralLinear(16 * cfg.ndf, 1)
        ]
        self._c_head = nn.Sequential(*_layers)

    def forward_u_head(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        u_predicts = self._u_head(x)
        return u_predicts

    def forward_c_head(self, x, c):
        feats = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        feats = torch.cat([feats, c], dim=1)
        c_predicts = self._c_head(feats)
        return c_predicts

    def forward(self, x, code=None):
        if code == None:
            score = self.forward_u_head(self._body(x))
        else:
            score = self.forward_c_head(self._body(x), code)
        return score


def discriminator_regularization(real_logits, real_input, fake_logits, fake_input):
    grad_real_logits = torch.autograd.grad(
        outputs=real_logits,
        inputs=real_input,
        grad_outputs=torch.ones_like(real_logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_fake_logits = torch.autograd.grad(
        outputs=fake_logits,
        inputs=fake_input,
        grad_outputs=torch.ones_like(fake_logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm_grad_real = grad_real_logits.view(grad_real_logits.size(0), -1).norm(2, dim=1) ** 2
    norm_grad_fake = grad_fake_logits.view(grad_real_logits.size(0), -1).norm(2, dim=1) ** 2

    real_pred = torch.sigmoid(real_logits)
    fake_pred = torch.sigmoid(fake_logits)

    reg_real = torch.square(1.0 - real_pred) * torch.square(norm_grad_real)
    reg_fake = torch.square(fake_pred) * torch.square(norm_grad_fake)
    reg_term = torch.mean(reg_real + reg_fake)
    return reg_term
