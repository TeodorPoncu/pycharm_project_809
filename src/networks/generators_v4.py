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
        target_size = cfg.ngf * 16
        style_size = self.latent_dim

        self.network = []
        self.scales = []
        while target_size != style_size:
            self.network += [nn.Sequential(*[nn.Linear(style_size, int(style_size/4)), nn.ReLU(inplace=True), nn.LayerNorm(int(style_size/4))])]
            style_size = int(style_size / 4)
            self.scales = self.scales + [style_size]

        while style_size != cfg.ngf * 2:
            self.network += [nn.Sequential(*[nn.Linear(style_size, int(style_size / 2)), nn.ReLU(inplace=True), nn.LayerNorm(int(style_size/2))])]
            style_size = int(style_size / 2)
            self.scales = self.scales + [style_size]

        self.network = nn.ModuleList(self.network)

    def forward(self, code):
        results = {}
        for idx, scale in enumerate(self.scales):
            if idx != 0:
                results[scale] = self.network[idx](results[self.scales[idx-1]])
            else:
                results[scale] = self.network[idx](code)
        return results

def AdaptiveNorm(input, style):
    size = input.size()
    x = input.view(size[0], size[1], size[2] * size[3])
    mean = x.mean(2, keepdim=True)
    x = x - mean
    std = torch.rsqrt((x ** 2).mean(2, keepdim=True) + 1e-5)
    norm_features = (x * std).view(*size)

    output = norm_features * style.unsqueeze(dim=-1).unsqueeze(dim=-1)
    return output

class StyleGenerator(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.latent_scale = int(cfg.load_size // (2 ** 6))
        self.latent_channels = int((cfg.latent_dim * 2) // (self.latent_scale ** 2))

        _layers = []
        self._init_conv = ConvBlock(self.latent_channels, cfg.ngf * 16, pad='zeros', act='relu', norm='instance', spectral=False)
        self._style_network = StyleNetwork(device, cfg)
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, act='relu', pad='reflect', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, act='relu', pad='reflect', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 16, act='relu', pad='reflect', spectral=False)]
        _layers = _layers + [UpsampleBlock(cfg.ngf * 16, cfg.ngf * 8, act='relu', pad='reflect', spectral=False)]
        self._trunk_layers = nn.ModuleList(_layers)

        _layers = [UpsampleBlock(cfg.ngf * 8, cfg.ngf * 4, pad='reflect', act='relu', spectral=False)]
        for _ in range(cfg.num_res):
            _layers = _layers + [ResBlock(cfg.ngf * 4, cfg.ngf * 4, pad='reflect', act='leaky_relu', norm='none', spectral=False)]
        self._stage_1_layers = nn.ModuleList(_layers)

        _layers = [UpsampleBlock(cfg.ngf * 4, cfg.ngf * 2, pad='reflect', act='relu', spectral=False)]
        for _ in range(cfg.num_res):
            _layers = _layers + [ResBlock(cfg.ngf * 2, cfg.ngf * 2, pad='reflect', act='leaky_relu', norm='none', spectral=False)]
        self._stage_2_layers = nn.ModuleList(_layers)

        self.heads = nn.ModuleList([ImageGenerator(device, cfg.ngf * factor, cfg) for factor in [8, 4, 2]])

    def forward(self, latent_vars, code):
        results = []
        style = self._style_network(latent_vars)
        features = code.view(code.size(0), self.latent_channels, self.latent_scale, self.latent_scale)
        features = self._init_conv(features)
        for idx in range(len(self._trunk_layers) -1):
            features = self._trunk_layers[idx](features)
            features = AdaptiveNorm(features, style[self.cfg.ngf * 16])

        features = self._trunk_layers[-1](features)
        features = AdaptiveNorm(features, style[self.cfg.ngf * 8])

        results = results + [self.heads[0](features)]

        for _layer in self._stage_1_layers:
            features = _layer(features)
            features = AdaptiveNorm(features, style[self.cfg.ngf * 4])

        results = results + [self.heads[1](features)]
        for _layer in self._stage_2_layers:
            features = _layer(features)
            features = AdaptiveNorm(features, style[self.cfg.ngf * 2])

        results = results + [self.heads[2](features)]

        return results

