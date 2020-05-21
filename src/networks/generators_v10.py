from ..layers.basic import EqualisedConvLayer, EqualisedLinearLayer, get_activ, get_norm, get_padding, spectral_norm, is_affine
from ..layers.normalization import InstanceNorm2d
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import transformers
import torch


class tRGB(nn.Module):
    def __init__(self, ic):
        super().__init__()
        self._transform = nn.Sequential(*[
            SpectralDecoupledConv(ic, 3, ks=3),
            nn.Tanh()
        ])

    def forward(self, x):
        return self._transform(x)


class SpectralDecoupledConv(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, bias=True, use_pad=True):
        super().__init__()
        self.stride = s
        self.dilation = d
        if use_pad:
            self.pad_size = int(np.ceil((ks - 1.0) / 2))
        else:
            self.pad_size = 0
        self._conv = spectral_norm(nn.Conv2d(ic, oc, ks, s, padding=self.pad_size, dilation=d, bias=bias))

    def get_bias(self):
        return self._conv.bias.view(1, -1, 1, 1)

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


class SpectralAdaptiveNorm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.std_layer = SpectralLinear(dim_in, dim_out)
        self.mean_layer = SpectralLinear(dim_in, dim_out)
        self.sequential_test = nn.Sequential()

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


class SpectralWeights(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        pass

class SpectralNoiseLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self._layer = spectral_norm(SpectralWeights(num_channels))
        self.noise = None

    def forward(self, x, noise=None):
        if self.noise == None and noise == None:
            noise = torch.rand(size=(x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        elif noise == None:
            noise = self.noise
        out = x + noise * self._layer.weight_orig.view(1, -1, 1, 1)
        return out


class StyleNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        global style_scales
        self.input_dim = cfg.latent_dim
        self._proj_dim = cfg.style_dim

        _style_net = [SpectralLinear(self.input_dim, self._proj_dim),
                      #nn.LeakyReLU(negative_slope=0.2),
                      # nn.BatchNorm1d(self._proj_dim)
                      ]
        for _ in range(cfg.style_layers):
            _style_net += [SpectralLinear(self._proj_dim, self._proj_dim),
                           #nn.LeakyReLU(negative_slope=0.2),
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
        # self.pad = get_padding(pad)(pad_size)

        self.k = min(ic, oc)
        self._conv_1 = SpectralDecoupledConv(ic, self.k, ks, bias=True)
        self._style_1 = SpectralAdaptiveNorm(cfg.style_dim, self.k)
        self._noise_1 = SpectralNoiseLayer(self.k)

        self._conv_2 = SpectralDecoupledConv(self.k, oc, ks, bias=True)
        self._style_2 = SpectralAdaptiveNorm(cfg.style_dim, oc)
        self._noise_2 = SpectralNoiseLayer(oc)

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
        # feat = self.pad(x)
        feat = x
        feat = self._conv_1(feat, add_bias=False)
        feat = self.act(feat)
        feat = self._style_1(style, feat)
        feat = self._noise_1(feat) + self._conv_1.get_bias()
        # feat = self.pad(feat)
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
        self._learn_channels = 16 * cfg.ngf - self.latent_channels
        self._const_init = torch.nn.Parameter(torch.rand(size=(self._learn_channels, self.latent_scale, self.latent_scale)))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        _init_layers = []
        for _ in range(1):
            _init_layers += [StyleBlock(cfg, 16 * cfg.ngf, 16 * cfg.ngf, upsample=True)]
        _init_layers += [StyleBlock(cfg, 16 * cfg.ngf, 16 * cfg.ngf, upsample=True)]
        _init_layers += [StyleBlock(cfg, 16 * cfg.ngf, 16 * cfg.ngf, upsample=True)]
        _init_layers += [StyleBlock(cfg, 16 * cfg.ngf, 16 * cfg.ngf, upsample=False)]
        self._init_layers = nn.ModuleList(_init_layers)

        generator_bodies = []
        generator_bodies.append(nn.ModuleList([StyleBlock(cfg, cfg.ngf * 16, cfg.ngf * 8, upsample=True),
                                               StyleBlock(cfg, cfg.ngf * 8, cfg.ngf * 8, upsample=False)]))
        generator_bodies.append(nn.ModuleList([StyleBlock(cfg, cfg.ngf * 8, cfg.ngf * 4, upsample=True),
                                               StyleBlock(cfg, cfg.ngf * 4, cfg.ngf * 4, upsample=False)]))
        generator_bodies.append(nn.ModuleList([StyleBlock(cfg, cfg.ngf * 4, cfg.ngf * 2, upsample=True),
                                               StyleBlock(cfg, cfg.ngf * 2, cfg.ngf * 2, upsample=False)]))
        self.generator_bodies = nn.ModuleList(generator_bodies)
        self.generator_heads = nn.ModuleList([tRGB(cfg.ngf * f) for f in [8, 4, 2]])
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
        for _layers, _head in zip(self.generator_bodies, self.generator_heads):
            for _l in _layers:
                feats, _ = _l(feats, style)
            results += [_head(feats)]
        return results
