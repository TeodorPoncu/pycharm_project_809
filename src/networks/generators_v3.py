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
        self.latent_channels = int((cfg.latent_dim * 2) // (self.latent_scale ** 2))

        _layers = []
        _layers = _layers + [ConvBlock(self.latent_channels, cfg.ngf * 16, pad='zeros', act='relu', norm='instance', spectral=False)]
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
        self._feat_dim = int(cfg.load_size // n_factor)
        self.latent_channels = int(cfg.latent_dim // (self._feat_dim ** 2))
        self._up_feats = None

        if self.latent_channels == 0:
            self.latent_channels = int(cfg.latent_dim // ((self._feat_dim / 2) ** 2))
            self._up_feats = UpsampleBlock(self.latent_channels, self.latent_channels, pad='reflect', act='relu', spectral=False)

        _layers = []
        _layers = _layers + [UpsampleBlock(cfg.ngf * factor, cfg.ngf * n_factor,
                                           pad='reflect', act='relu', norm='instance', spectral=False)]

        for _ in range(cfg.num_res):
            _layers = _layers + [ResBlock(cfg.ngf * n_factor, cfg.ngf * n_factor, pad='reflect', act='relu', norm='instance', spectral=False)]

        self._proj_layer = GLUConvBlock(cfg.ngf * factor + self.latent_channels, cfg.ngf * factor, pad='reflect', norm='instance', spectral=False)
        self._layers = nn.Sequential(*_layers)

    def forward(self, x, c):
        if self._up_feats is not None:
            c = c.view(c.size(0), self.latent_channels, int(self._feat_dim/2), int(self._feat_dim/2))
            c = self._up_feats(c)
        else:
            c = c.view(c.size(0), self.latent_channels, self._feat_dim, self._feat_dim)
        x = torch.cat([x, c], dim=1)
        x = self._proj_layer(x)
        return self._layers(x)


