import torch
import torch.nn as nn
from ..layers.basic import *
from ..layers.normalization import InstanceNorm2d


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).requires_grad_(False)

    def __call__(self, inputs, target_is_real=True):
        print(inputs.size(), inputs, target_is_real)
        if target_is_real:
            values = torch.min(-1 + inputs, self.get_zero_tensor(inputs))
            loss = -torch.mean(values)
        else:
            values = torch.min(-1 - inputs, self.get_zero_tensor(inputs))
            loss = -torch.mean(values)

        return loss

class HingeJointConvolutionalDiscriminator(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.real_label = (1 - cfg.smooth) * cfg.target_real_label
        self.fake_label = (1 - cfg.smooth) * cfg.target_fake_label
        self._u_head = None
        self._c_head = None
        self.criterion_d = HingeLoss()

        self._build_feature_layers(cfg)
        self._build_u_head(cfg)

    def _build_feature_layers(self, cfg):
        _layers = []
        _layers = _layers + [ConvBlock(ic=3, oc=cfg.ndf, ks=7, pad='reflect', act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]

        f = 1
        pf = 1

        for _ in range(cfg.dsc_layers):
            f = min(8, f * 2)
            _layers = _layers + [ConvBlock(ic=cfg.ndf * pf, oc=cfg.ndf * f, ks=3, s=2, pad='reflect', act='leaky_relu', norm='instance',spectral=cfg.dsc_spectral)]
            pf = f

        self.feature_extractor = nn.Sequential(*_layers)
        self._factor = f

    def _build_c_head(self, cfg, scale):
        self._text_channels = int(cfg.latent_dim // scale)
        _c_head = [ConvBlock(ic=cfg.ndf * self._factor + self._text_channels, oc=cfg.ndl, ks=3, pad='reflect',
                             act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]
        _c_head = _c_head + [nn.AdaptiveAvgPool2d(1)]
        _c_head = _c_head + [LinearBlock(cfg.ndl, 1, bias=False, act='leaky_relu', spectral=False)]
        self._c_head = nn.ModuleList(_c_head).to(self.device)

    def _build_u_head(self, cfg):
        _u_head = [ConvBlock(ic=cfg.ndf * self._factor, oc=cfg.ndl, ks=3, pad='reflect',
                             act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]
        _u_head = _u_head + [nn.AdaptiveAvgPool2d(1)]
        _u_head = _u_head + [LinearBlock(cfg.ndl, 1, bias=False, act='leaky_relu', spectral=False)]
        self._u_head = nn.ModuleList(_u_head)

    def forward_features(self, x):
        features = self.feature_extractor(x)
        return features

    def forward_u_head(self, x):
        u_features = self._u_head[0](x)
        u_features = self._u_head[1](u_features)
        u_features = u_features.view(-1, self.cfg.ndl)
        u_scores = self._u_head[2](u_features)
        return u_scores

    def forward_c_head(self, x, c):
        features = x
        if self._c_head is None:
            scale = x.size(2) * x.size(3)
            self._build_c_head(self.cfg, scale)
        c_features = c.view(c.size(0), self._text_channels, features.size(2), features.size(3))
        c_features = torch.cat([features, c_features], dim=1)
        c_features = self._c_head[0](c_features)
        c_features = self._c_head[1](c_features)
        c_features = c_features.view(-1, self.cfg.ndl)
        c_scores = self._c_head[2](c_features)
        return c_scores

    def forward(self, x, c):
        features = self.forward_features(x)
        return {'u': self.forward_u_head(features), 'c': self.forward_c_head(features, c)}

    def compute_loss(self, score_real, score_fake):
        loss = {}
        #score_fake = self.forward(fake.detach(), c)
        #score_real = self.forward(real, c)
        loss['u_real'] = self.criterion_d(score_real['u'], target_is_real=True)
        loss['c_real'] = self.criterion_d(score_real['c'], target_is_real=True)
        loss['u_fake'] = self.criterion_d(score_fake['u'], target_is_real=False)
        loss['c_fake'] = self.criterion_d(score_fake['c'], target_is_real=False)
        return loss