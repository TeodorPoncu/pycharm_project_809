from ..layers.basic import EqualisedConvLayer, EqualisedLinearLayer, get_activ, get_norm, get_padding, spectral_norm, is_affine
from ..layers.normalization import InstanceNorm2d
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import transformers
import torch




class tRGB(nn.Module):
    def __init__(self, ic, custom_lr=1., weight_scaling=True):
        super().__init__()
        self._transform = nn.Sequential(*[
            (ic, ic, ks=3, weight_scaling=weight_scaling, custom_lr=custom_lr),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=ic),
            EqualisedConvLayer(ic, 3, ks=1, weight_scaling=weight_scaling, custom_lr=custom_lr),
            nn.Tanh()
        ])

    def forward(self, input):
        return self._transform(input)


class AdaptiveStyleLayer(nn.Module):
    def __init__(self, input_size, num_features, custom_lr=1., weight_scaling=True):
        super().__init__()
        # mapping layers are included here for simplicity
        self.std_layer = EqualisedLinearLayer(input_size, num_features, bias=True, custom_lr=custom_lr, weight_scaling=weight_scaling)
        self.mean_layer = EqualisedLinearLayer(input_size, num_features, bias=True, custom_lr=custom_lr, weight_scaling=weight_scaling)

    def forward(self, features, x):
        s_mean = self.mean_layer(features)
        s_std = self.std_layer(features)

        size = x.size()
        x = x.view(size[0], size[1], size[2] * size[3])
        mean = x.mean(2, keepdim=True)
        x = x - mean
        std = torch.rsqrt((x ** 2).mean(2, keepdim=True) + 1e-8)
        norm_features = (x * std).view(*size)
        output = norm_features * s_std.unsqueeze(-1).unsqueeze(-1) + s_mean.unsqueeze(-1).unsqueeze(-1)
        return output


class NoiseLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_channels))
        self.noise = None

    def forward(self, x, noise=None):
        if self.noise == None and noise == None:
            noise = torch.rand(size=(x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        elif noise == None:
            noise = self.noise
        out = x + noise * self.weight.view(1, -1, 1, 1)
        return out


class StyleNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        global style_scales
        self.input_dim = cfg.latent_dim
        self._proj_dim = cfg.style_dim

        _style_net = [EqualisedLinearLayer(self.input_dim, self._proj_dim, bias=cfg.style_bias, custom_lr=cfg.style_lr_mul,
                                           weight_scaling=cfg.weight_scaling),
                      nn.LeakyReLU(negative_slope=0.2),
                      # nn.BatchNorm1d(self._proj_dim)
                      ]
        for _ in range(cfg.style_layers):
            _style_net += [EqualisedLinearLayer(self._proj_dim, self._proj_dim, bias=cfg.style_bias, custom_lr=cfg.style_lr_mul,
                                                weight_scaling=cfg.weight_scaling),
                           nn.LeakyReLU(negative_slope=0.2),
                           # nn.BatchNorm1d(self._proj_dim)
                           ]
        self._style_net = nn.Sequential(*_style_net)

    def forward(self, x):
        style = self._style_net(x)
        return style


class StyleBlock(nn.Module):
    def __init__(self, cfg, ic, oc, ks=3, pad='reflect', act='leaky_relu', upsample=False):
        global style_scales
        super().__init__()

        pad_size = int(np.ceil((ks - 1.0) / 2))
        self.act = get_activ(act)()
        self.pad = get_padding(pad)(pad_size)

        self.k = min(ic, oc)
        self._conv_1 = EqualisedConvLayer(ic, self.k, ks, bias=True, weight_scaling=cfg.weight_scaling, custom_lr=1.)
        self._style_1 = AdaptiveStyleLayer(cfg.style_dim, self.k, weight_scaling=cfg.weight_scaling, custom_lr=cfg.style_lr_mul)
        self._noise_1 = NoiseLayer(self.k)

        self._conv_2 = EqualisedConvLayer(self.k, oc, ks, bias=True, weight_scaling=cfg.weight_scaling, custom_lr=1.)
        self._style_2 = AdaptiveStyleLayer(cfg.style_dim, oc, weight_scaling=cfg.weight_scaling, custom_lr=cfg.style_lr_mul)
        self._noise_2 = NoiseLayer(oc)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.upsample = None

    def forward(self, input, style):
        if isinstance(input, tuple):
            in_feat, _ = input
        else:
            in_feat = input
        if self.upsample:
            x = self.upsample(in_feat)
        else:
            x = in_feat
        feat = self.pad(x)
        feat = self._conv_1(feat, add_bias=False)
        feat = self.act(feat)
        feat = self._style_1(style, feat)
        feat = self._noise_1(feat) + self._conv_1.get_bias()
        feat = self.pad(feat)
        feat = self._conv_2(feat, add_bias=False)
        feat = self.act(feat)
        feat = self._style_2(style, feat)
        feat = self._noise_2(feat) + self._conv_2.get_bias()
        return feat, x


class StyleGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        global style_scales
        style_scales = []
        self.cfg = cfg
        self.latent_scale = int(cfg.load_size // (2 ** 6))
        self.latent_channels = int(cfg.latent_dim) // (self.latent_scale ** 2)
        self._learn_channels = 8 * cfg.ngf - self.latent_channels
        self._const_init = torch.nn.Parameter(torch.rand(size=(self._learn_channels, self.latent_scale, self.latent_scale)))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        _init_layers = []
        for _ in range(1):
            _init_layers += [StyleBlock(cfg, 8 * cfg.ngf, 8 * cfg.ngf, upsample=True)]
        _init_layers += [StyleBlock(cfg, 8 * cfg.ngf, 8 * cfg.ngf, upsample=True)]
        _init_layers += [StyleBlock(cfg, 8 * cfg.ngf, 8 * cfg.ngf, upsample=True)]
        _init_layers += [StyleBlock(cfg, 8 * cfg.ngf, 8 * cfg.ngf, upsample=False)]
        self._init_layers = nn.ModuleList(_init_layers)

        generator_bodies = []
        generator_bodies += nn.ModuleList([StyleBlock(cfg, cfg.ngf * 8, cfg.ngf * 4, upsample=True),
                                           StyleBlock(cfg, cfg.ngf * 4, cfg.ngf * 4, upsample=False)])
        generator_bodies += nn.ModuleList([StyleBlock(cfg, cfg.ngf * 4, cfg.ngf * 2, upsample=True),
                                           StyleBlock(cfg, cfg.ngf * 2, cfg.ngf * 2, upsample=False)])
        generator_bodies += nn.ModuleList([StyleBlock(cfg, cfg.ngf * 2, cfg.ngf * 1, upsample=True),
                                           StyleBlock(cfg, cfg.ngf * 1, cfg.ngf * 1, upsample=False)])
        self.generator_bodies = nn.ModuleList(generator_bodies)
        self.generator_heads = nn.ModuleList([tRGB(cfg.ngf * f, weight_scaling=cfg.weight_scaling) for f in [4, 2, 1]])

        self.style_network = StyleNetwork(cfg)

    def forward(self, input_style, input_content):
        content = input_content.view(input_content.size(0), self.latent_channels, self.latent_scale, self.latent_scale)
        const = self._const_init.repeat(input_content.size(0), 1, 1, 1)
        feats = torch.cat([content, const], dim=1)
        style = self.style_network(input_style)
        for _layer in self._init_layers:
            feats = _layer(feats, style)

        # TODO: add style mixing for v9 (randomly sample between a style, and interpolated style)
        # TODO: should do something with those noise, layers as well
        # TODO: for style mixing, use fixed noisem
        results = []
        for _layer, _head in zip(self.generator_bodies, self.generator_heads):
            feats, _ = _layer(feats, style)
            results += [_head(feats)]
        return results
