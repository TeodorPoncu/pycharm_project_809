from ..layers.basic import LinearBlock, get_activ, get_norm, get_padding, spectral_norm, is_affine
import numpy as np
import torch.nn as nn
import torch

style_scales = []

class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, s=1, d=1, pad='none', act='none', norm='none', bias=False, spectral=True):
        super().__init__()
        self.spectral = spectral
        self.activation = get_activ(act)
        self.normalization = get_norm(norm)
        self.padding = get_padding(pad)
        self.bias = bias
        self._build_layers(ic, oc, ks, s, d)

    def _build_layers(self, ic, oc, ks, s, d):
        _layers = []
        pad_size = int(np.ceil((ks - 1.0) / 2))
        if self.padding is not None:
            _layers = _layers + [self.padding(pad_size)]
        if self.spectral:
            _layers = _layers + [spectral_norm(nn.Conv2d(ic, oc, ks, s, 0, d, bias=self.bias))]
        elif not self.spectral:
            _layers = _layers + [nn.Conv2d(ic, oc, ks, s, 0, d, bias=self.bias)]
        if self.activation is not None:
            _layers = _layers + [self.activation()]
        if self.normalization is not None:
            if self.spectral and is_affine(self.normalization):
                _layers = _layers + [spectral_norm(self.normalization(oc))]
            _layers = _layers + [self.normalization(oc)]
        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv(x)
        return x


def AdaptiveNorm(x, style):
    size = x.size()
    x = x.view(size[0], size[1], size[2] * size[3])
    mean = x.mean(2, keepdim=True)
    x = x - mean
    std = torch.rsqrt((x ** 2).mean(2, keepdim=True) + 1e-5)
    norm_features = (x * std).view(*size)

    output = norm_features * style.unsqueeze(dim=-1).unsqueeze(dim=-1)
    return output

class RNN(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.hidden_dim = cfg.rnn_hidden_dim
        self.hidden_layers = cfg.rnn_num_layers
        self._wte = nn.Embedding(vocab_size, cfg.embedding_size)
        self._net = nn.LSTM(cfg.embedding_size, self.hidden_dim, self.hidden_layers, dropout=0.2, batch_first=False, bidirectional=True)
        self._init_h = nn.Parameter(torch.zeros(self.hidden_layers * 2, cfg.batch_size, self.hidden_dim).type(torch.FloatTensor), requires_grad=True)
        self._init_c = nn.Parameter(torch.zeros(self.hidden_layers * 2, cfg.batch_size, self.hidden_dim).type(torch.FloatTensor), requires_grad=True)

    def forward(self, pad_ids, len_ids):
        emb = self._wte(pad_ids)
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, len_ids, batch_first=False, enforce_sorted=False)
        out, (_, hid) = self._net(emb, hx=(self._init_c, self._init_h))
        hid = hid.permute(1, 0, 2)
        hid = hid.view(hid.size(0), self.hidden_layers, 2, hid.size(2))
        hid = hid[:, -1, :, :]
        fwd, bwd = hid.split(1, dim=1)
        fwd = fwd.view(hid.size(0), -1)
        bwd = bwd.view(bwd.size(0), -1)
        hid = torch.cat([bwd, fwd], dim=1)
        return hid

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

        _style_net = [LinearBlock(self.input_dim, self._proj_dim, act='none', norm='none', spectral=cfg.gen_spectral)]
        for _ in range(cfg.style_layers):
            _style_net += [LinearBlock(self._proj_dim, self._proj_dim, act='none', norm='none', spectral=cfg.gen_spectral)]
        self._style_net = nn.Sequential(*_style_net)

        _mapping_net = []
        for scale in style_scales:
            _mapping_net += [LinearBlock(self._proj_dim, scale, act='none', norm='none', spectral=cfg.gen_spectral)]
        self._mapping_net = nn.ModuleList(_mapping_net)

    def forward(self, x):
        style = self._style_net(x)
        style_maps = []
        for idx in range(0, len(style_scales), 2):
            style_maps += [(self._mapping_net[idx](style), self._mapping_net[idx + 1](style))]
        return style_maps

class StyleBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, pad='reflect', act='relu', upsample=False, spectral=False):
        global style_scales
        super().__init__()
        self.k = min(ic, oc)
        self._conv_1 = ConvBlock(ic, self.k, ks, pad=pad, act=act, spectral=spectral)
        self._noise_1 = NoiseLayer(self.k)
        self._conv_2 = ConvBlock(self.k, oc, ks, pad=pad, act=act, spectral=spectral)
        self._noise_2 = NoiseLayer(oc)

        style_scales += [self.k, oc]
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample = None

    def forward(self, input, style_1, style_2):
        if self.upsample:
            x = self.upsample(input)
        else:
            x = input
        feat = self._conv_1(x)
        feat = self._noise_1(feat)
        feat = AdaptiveNorm(feat, style_1)
        feat = self._conv_2(feat)
        feat = self._noise_2(feat)
        feat = AdaptiveNorm(feat, style_2)
        return feat, x


class StyleResBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, pad='reflect', act='relu', upsample=False, spectral=False):
        super().__init__()
        self.fwd_block = StyleBlock(ic, oc, ks, pad, act, upsample)
        if ic != oc:
            self.res_block = ConvBlock(ic, oc, ks, pad=pad, act=act, spectral=spectral)
        else:
            self.res_block = None

    def forward(self, input, style_1, style_2):
        block_out, block_inp = self.fwd_block(input, style_1, style_2)
        res = block_inp
        if self.res_block:
            res = self.res_block(block_inp)
            res = AdaptiveNorm(res, style_2)
        return block_out + res

class GeneratorTrunk(nn.Module):
    def __init__(self, ic, oc, num_block, spectral):
        super().__init__()
        _layers = []
        if ic == oc:
            for _ in range(num_block):
                   _layers += [StyleResBlock(ic, oc, spectral=spectral)]
        else:
            _layers  += [StyleResBlock(ic, oc, upsample=True, spectral=spectral)]
            for _ in range(num_block - 1):
                _layers += [StyleResBlock(oc, oc, upsample=False, spectral=spectral)]
        self._layers = nn.ModuleList(_layers)

    def forward(self, x, styles):
        for (_layer, style_pair) in zip(self._layers, styles):
            style_1, style_2 = style_pair
            x = _layer(x, style_1, style_2)
        return x

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
        self.spectral = cfg.gen_spectral
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        _init_layers = []
        for _ in range(1):
            _init_layers += [StyleBlock(16 * cfg.ngf, 16 * cfg.ngf, upsample=True, spectral=cfg.gen_spectral)]
        _init_layers += [StyleBlock(16 * cfg.ngf, 16 * cfg.ngf, upsample=True, spectral=cfg.gen_spectral)]
        _init_layers += [StyleBlock(16 * cfg.ngf, 8 * cfg.ngf, upsample=True, spectral=cfg.gen_spectral)]
        _init_layers += [StyleBlock(8 * cfg.ngf, 8 * cfg.ngf, upsample=False, spectral=cfg.gen_spectral)]
        self._init_layers = nn.ModuleList(_init_layers)

        generator_bodies = []
        generator_bodies += [GeneratorTrunk(cfg.ngf * 8, cfg.ngf * 4, num_block=3, spectral=cfg.gen_spectral)]
        generator_bodies += [GeneratorTrunk(cfg.ngf * 4, cfg.ngf * 2, num_block=3, spectral=cfg.gen_spectral)]
        generator_bodies += [GeneratorTrunk(cfg.ngf * 2, cfg.ngf * 1, num_block=3, spectral=cfg.gen_spectral)]
        self.generator_bodies = nn.ModuleList(generator_bodies)
        self.generator_heads = nn.ModuleList([ConvBlock(cfg.ngf * f, 3, pad='reflect', act='tanh', bias=True, spectral=self.spectral) for f in [4, 2, 1]])

        self.style_network = StyleNetwork(cfg)

    def forward(self, input_style, input_content):
        style_pairs = self.style_network(input_style)
        content = input_content.view(input_content.size(0), self.latent_channels, self.latent_scale, self.latent_scale)
        const = self._const_init.repeat(input_content.size(0), 1, 1, 1)
        feats = torch.cat([content, const], dim=1)
        idx = 0
        for _layer in self._init_layers:
            style_1, style_2 = style_pairs[idx]
            feats, _ = _layer(feats, style_1, style_2)
            idx = idx + 1

        results = []
        for _layer, _head in zip(self.generator_bodies, self.generator_heads):
            feats = _layer(feats, style_pairs[idx:idx+3])
            results += [_head(feats)]
            idx = idx + 3

        return results