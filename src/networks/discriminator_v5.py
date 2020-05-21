import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.basic import *
from ..layers.normalization import InstanceNorm2d

class ViewLayer(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.real_label = cfg.real_label - cfg.smooth
        self.fake_label = cfg.fake_label
        self.generator_label = cfg.real_label
        self.loss = nn.MSELoss()
        self.real_label_tensor = None
        self.fake_label_tensor = None
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
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def __call__(self, input, condition=None, condition_similarity=1, target_is_real=True, for_discriminator=True):
        target_tensor = self.get_target_tensor(input, target_is_real, for_discriminator)
        loss = self.loss(input, target_tensor.to(input.device))
        return loss


class PatchConvolutionalDiscriminator(nn.Module):
    def __init__(self, input_dim, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.criterion = BinaryCrossEntropyLoss(cfg)
        self._build_feature_layers(input_dim, cfg)
        self._build_u_head(cfg)
        self._build_c_head(cfg)

    def _build_feature_layers(self, input_dim, cfg):
        _layers = []
        _layers = _layers + [ConvBlock(ic=3, oc=cfg.ndf, ks=7, pad='reflect', act='leaky_relu', norm='batch', spectral=cfg.dsc_spectral)]
        _conv_dim = input_dim
        self.input_dim = input_dim
        self.init = input_dim
        f = 1
        pf = 1

        while self.input_dim != 8:
            f = min(8, f * 2)
            _layers = _layers + [ConvBlock(ic=cfg.ndf * pf, oc=cfg.ndf * f, ks=3, s=2, pad='reflect', act='leaky_relu', norm='batch',
                                           spectral=cfg.dsc_spectral)]
            pf = f
            self.input_dim = int(self.input_dim/2)

        self.feature_extractor = nn.Sequential(*_layers)
        self._factor = f

    def _build_u_head(self, cfg):
        _u_head=  []
        _u_head = _u_head + [ConvBlock(ic=cfg.ndf * self._factor, oc=cfg.ndf * self._factor, ks=3, pad='reflect',
                             act='leaky_relu', norm='batch', spectral=cfg.dsc_spectral)]
        _u_head += [nn.AvgPool2d(kernel_size=(2, 2), stride=2)]
        _u_head = _u_head + [ViewLayer(-1)]
        _u_head = _u_head + [LinearBlock(16 * cfg.ndf * self._factor, cfg.ndf * self._factor, bias=True, act='leaky_relu')]
        _u_head = _u_head + [ViewLayer(-1)]
        _u_head = _u_head + [nn.Linear(cfg.ndf * self._factor, 1)]
        self._u_head = nn.Sequential(*_u_head)

    def _build_c_head(self, cfg):
        self._latent_channels = int(cfg.latent_dim / 64)
        _c_head = []
        _c_head = _c_head + [ConvBlock(ic=cfg.ndf * self._factor + self._latent_channels, oc=cfg.ndf * self._factor, ks=3, pad='reflect',
                                       act='leaky_relu', norm='batch', spectral=cfg.dsc_spectral)]
        _c_head += [nn.AvgPool2d(kernel_size=(2, 2), stride=2)]
        _c_head = _c_head + [ViewLayer(-1)]
        _c_head = _c_head + [LinearBlock(16 * cfg.ndf * self._factor, cfg.ndf * self._factor, bias=True, act='leaky_relu')]
        _c_head = _c_head + [ViewLayer(-1)]
        _c_head = _c_head + [nn.Linear(cfg.ndf * self._factor, 1)]
        self._c_head = nn.Sequential(*_c_head)

    def forward_u_head(self, x):
        u_predicts = self._u_head(x)
        return u_predicts

    def forward_c_head(self, x, c):
        c = c.view(x.size(0), self._latent_channels, 8, 8)
        c = torch.cat([x, c], dim=1)
        c_predicts = self._c_head(c)
        return c_predicts

    def forward_features(self, x):
        features = self.feature_extractor(x)
        return features

    def forward(self, x, code=None):
        feats =  self.forward_features(x)
        if code == None:
            score = self.forward_u_head(feats)
        else:
            score = self.forward_c_head(feats, code)
        return score
