from ..layers.basic import *
import torch.nn as nn
import torch


class ImageGenerator(nn.Module):
    def __init__(self, device, in_channels, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device

        _layers = []
        _layers = _layers + [ConvBlock(in_channels, cfg.ngf, pad='reflect', act='relu', norm='instance', spectral=False)]
        _layers = _layers + [ConvBlock(cfg.ngf, 3, pad='reflect', act='tanh', norm='none', spectral=False)]

        self._layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self._layers(x)


class VanillaGenerator(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.latent_scale = int(cfg.load_size // (2 ** 6))
        self.latent_channels = int(cfg.latent_dim // (self.latent_scale ** 2))

        _layers = []
        _layers = _layers + [GLUConvBlock(self.latent_channels, cfg.ngf * 16, pad='zeros', norm='instance', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, pad='reflect', act='relu', norm='instance', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, pad='reflect', act='relu', norm='instance', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, pad='reflect', act='relu', norm='instance', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 8, pad='reflect', act='relu', norm='instance', spectral=False)]

        self._layers = nn.Sequential(*_layers)

    def forward(self, x):
        reshaped_x = x.view(-1, self.latent_channels, self.latent_scale, self.latent_scale)
        return self._layers(reshaped_x)


class ResidualGenerator(nn.Module):
    def __init__(self, device, cfg, factor):
        super().__init__()
        self.cfg = cfg
        self.device = device
        n_factor = int(factor / 2)
        _text_channels = int(cfg.latent_dim // 64)
        _feat_dim = int(cfg.load_size // n_factor)
        _c_dim = 8

        _layers = []
        _c_up_head = []
        while _c_dim <= _feat_dim:
            _c_up_head = _c_up_head + [
                UpsampleBlock(_text_channels, _text_channels, ks=1, act='leaky_relu', norm='instance', spectral=cfg.dsc_spectral)]
            _feat_dim = _feat_dim * 2

        _layers = _layers + [GLUConvBlock(cfg.ngf * factor + _text_channels, cfg.ngf * factor, spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * factor, cfg.ngf * n_factor,
                                           pad='reflect', act='relu', norm='instance', spectral=False)]


        for _ in range(cfg.num_res):
            _layers = _layers + [GLUResBlock(cfg.ngf * n_factor, cfg.ngf * n_factor, pad='reflect', norm='instance', spectral=False)]

        self._c_up_head = nn.Sequential(*_c_up_head)
        self._layers = nn.Sequential(*_layers)

    def forward(self, x, c):
        return self._layers(x)


