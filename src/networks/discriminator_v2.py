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
        if target_is_real:
            values = torch.min(-1 + inputs, self.get_zero_tensor(inputs))
            loss = -torch.mean(values)
        else:
            values = torch.min(-1 - inputs, self.get_zero_tensor(inputs))
            loss = -torch.mean(values)

        return loss


class HingeJointPatchConvolutionalDiscriminator(nn.Module):
    def __init__(self, input_dim, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.real_label = (1 - cfg.smooth) * cfg.target_real_label
        self.fake_label = (1 - cfg.smooth) * cfg.target_fake_label
        self._u_head = None
        self._c_head = None
        self.criterion_d = HingeLoss()

        self._build_feature_layers(input_dim, cfg)
        self._build_u_head(cfg)
        self._build_c_head(cfg)

    def _build_feature_layers(self, input_dim, cfg):
        _layers = []
        _layers = _layers + [ConvBlock(ic=3, oc=cfg.ndf, ks=7, pad='reflect', act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]
        _conv_dim = input_dim
        self.input_dim = input_dim
        self.init = input_dim
        f = 1
        pf = 1

        while self.input_dim != 8:
            f = min(8, f * 2)
            _layers = _layers + [ConvBlock(ic=cfg.ndf * pf, oc=cfg.ndf * f, ks=3, s=2, pad='reflect', act='leaky_relu', norm='instance',
                                           spectral=cfg.dsc_spectral)]
            pf = f
            self.input_dim = int(self.input_dim/2)

        self.feature_extractor = nn.Sequential(*_layers)
        self._factor = f

    def _build_c_head(self, cfg):
        _c_up_head = []
        _c_head = []
        self._text_channels = int(cfg.latent_dim // 64)
        while self._text_channels != cfg.ndf * self._factor:
            _c_up_head = _c_up_head + [
                ConvBlock(self._text_channels, self._text_channels * 2, ks=1, act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]
            self._text_channels = self._text_channels * 2
        _c_head = _c_head + [nn.GLU(dim=1)]
        _c_head = _c_head + [ConvBlock(ic=cfg.ndf * self._factor, oc=cfg.ndl, ks=3, pad='reflect',
                             act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]
        _c_head = _c_head + [ConvBlock(ic=cfg.ndl, oc=1, ks=1, pad='none',
                             act='leaky_relu', norm='none', spectral=cfg.dsc_spectral)]
        self._c_up_head = nn.Sequential(*_c_up_head)
        self._c_head = nn.Sequential(*_c_head)

    def _build_u_head(self, cfg):
        _u_head = [nn.GLU(dim=1)]
        _u_head = _u_head + [ConvBlock(ic=cfg.ndf * self._factor, oc=cfg.ndl, ks=3, pad='reflect',
                             act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]
        _u_head = _u_head + [ConvBlock(ic=cfg.ndl, oc=1, ks=1, pad='none',
                                       act='leaky_relu', norm='none', spectral=cfg.dsc_spectral)]
        self._u_proj = ConvBlock(ic=cfg.ndf * self._factor, oc=cfg.ndf * self._factor, ks=1, pad='none', act='leaky_relu', norm='instance',
                                 spectral=cfg.dsc_spectral)
        self._u_head = nn.Sequential(*_u_head)

    def forward_features(self, x):
        features = self.feature_extractor(x)
        return features

    def forward_u_head(self, x):
        u_sig_features = self._u_proj(x)
        u_features = torch.cat([x, u_sig_features], dim=1)
        u_predicts = self._u_head(u_features)
        return u_predicts

    def forward_c_head(self, features, c):
        c = c.view(c.size(0), -1, 8, 8)
        c_features = self._c_up_head(c)
        c_features = torch.cat([features, c_features], dim=1)
        c_predicts = self._c_head(c_features)
        return c_predicts

    def forward(self, x, c):
        features = self.forward_features(x)
        return {'u': self.forward_u_head(features), 'c': self.forward_c_head(features, c)}

    def compute_loss(self, score_real, score_fake):
        loss = {}
        # score_fake = self.forward(fake.detach(), c)
        # score_real = self.forward(real, c)
        loss['u_real'] = self.criterion_d(score_real['u'], target_is_real=True)
        loss['c_real'] = self.criterion_d(score_real['c'], target_is_real=True)
        loss['u_fake'] = self.criterion_d(score_fake['u'], target_is_real=False)
        loss['c_fake'] = self.criterion_d(score_fake['c'], target_is_real=False)
        return loss
