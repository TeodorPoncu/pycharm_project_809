from ..layers.basic import *
import torch.nn as nn
import torch


def AdaptiveNorm(x, style):
    size = x.size()
    x = x.view(size[0], size[1], size[2] * size[3])
    mean = x.mean(2, keepdim=True)
    x = x - mean
    std = torch.rsqrt((x ** 2).mean(2, keepdim=True) + 1e-5)
    norm_features = (x * std).view(*size)

    output = norm_features * style.unsqueeze(dim=-1).unsqueeze(dim=-1)
    return output


class ImageGenerator(nn.Module):
    def __init__(self, device, in_channels, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device

        _layers = []
        _layers = _layers + [ConvBlock(in_channels, cfg.ngf, pad='reflect', act='relu', norm='instance', spectral=False)]
        _layers = _layers + [ConvBlock(cfg.ngf, 3, ks=7, pad='reflect', act='tanh', norm='none', spectral=False)]

        self._layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self._layers(x)


class StyleNetwork(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.project_size = 128
        style_size = self.latent_dim

        _layers = [nn.Sequential(*[nn.Linear(style_size, int(style_size / 8)),
                                   nn.LeakyReLU(inplace=True),
                                   nn.LayerNorm(int(style_size / 8))])]
        for _ in range(cfg.style_layers - 1):
            _layers += [
                nn.Sequential(*[nn.Linear(int(style_size / 8), int(style_size / 8)),
                                nn.LeakyReLU(inplace=True),
                                nn.LayerNorm(int(style_size / 8))])]

        self._style_network = nn.Sequential(*_layers)
        self._style_projections = {}
        for factor in [1, 2, 3, 4]:
            scale_v = int((2 ** factor) * cfg.ngf)
            scale_k = str(scale_v)
            self._style_projections[scale_k] = nn.Sequential(*[nn.Linear(int(style_size / 8), scale_v),
                                                               nn.ReLU(inplace=True),
                                                               nn.LayerNorm(scale_v)])
        self._style_projections = nn.ModuleDict(self._style_projections)

    def forward(self, code):
        results = {}
        style = self._style_network(code)
        for scale in self._style_projections:
            results[scale] = self._style_projections[scale](style)
        return results


class StyleGenerator(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.latent_scale = int(cfg.load_size // (2 ** 6))
        self.latent_channels = int((cfg.latent_dim * 2) // (self.latent_scale ** 2))

        _layers = []
        self._init_conv = ConvBlock(self.latent_channels, cfg.ngf * 16, pad='zeros', act='leaky_relu', norm='none', spectral=False)
        self._style_network = StyleNetwork(device, cfg)
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, act='leaky_relu', pad='reflect', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, act='leaky_relu', pad='reflect', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, act='leaky_relu', pad='reflect', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 8, act='leaky_relu', pad='reflect', spectral=False)]
        self._trunk_layers = nn.ModuleList(_layers)

        _layers = [UpsampleBlock(cfg.ngf * 8, cfg.ngf * 4, pad='reflect', act='leaky_relu', spectral=False)]
        for _ in range(cfg.num_res):
            _layers = _layers + [ResBlock(cfg.ngf * 4, cfg.ngf * 4, pad='reflect', act='leaky_relu', norm='none', spectral=False)]
        self._stage_1_layers = nn.ModuleList(_layers)

        _layers = [UpsampleBlock(cfg.ngf * 4, cfg.ngf * 2, pad='reflect', act='leaky_relu', spectral=False)]
        for _ in range(cfg.num_res):
            _layers = _layers + [ResBlock(cfg.ngf * 2, cfg.ngf * 2, pad='reflect', act='leaky_relu', norm='none', spectral=False)]
        self._stage_2_layers = nn.ModuleList(_layers)

        self.heads = nn.ModuleList([ImageGenerator(device, cfg.ngf * factor, cfg) for factor in [8, 4, 2]])

    def forward(self, latent_vars, code):
        results = []
        style = self._style_network(latent_vars)
        features = code.view(code.size(0), self.latent_channels, self.latent_scale, self.latent_scale)
        features = self._init_conv(features)
        for idx in range(len(self._trunk_layers) - 1):
            features = self._trunk_layers[idx](features)
            features = AdaptiveNorm(features, style[str(self.cfg.ngf * 16)])
            features = features + torch.randn_like(features)

        features = self._trunk_layers[-1](features)
        features = AdaptiveNorm(features, style[str(self.cfg.ngf * 8)])
        features = features + torch.randn_like(features)

        results = results + [self.heads[0](features)]

        for _layer in self._stage_1_layers:
            features = _layer(features)
            features = AdaptiveNorm(features, style[str(self.cfg.ngf * 4)])
            features = features + torch.randn_like(features)

        results = results + [self.heads[1](features)]
        for _layer in self._stage_2_layers:
            features = _layer(features)
            features = AdaptiveNorm(features, style[str(self.cfg.ngf * 2)])
            features = features + torch.randn_like(features)

        results = results + [self.heads[2](features)]

        return results
