from ..layers.basic import EqualisedConvLayer, EqualisedLinearLayer, get_activ, get_norm, get_padding, spectral_norm, is_affine
from ..layers.normalization import InstanceNorm2d
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class AdaptiveStyleLayer(nn.Module):
    def __init__(self, input_size, num_features, custom_lr=1., weight_scaling=True):
        super().__init__()
        # mapping layers are included here for simplicity
        self.std_layer = EqualisedLinearLayer(input_size, num_features, bias=True,
                                                 custom_lr=custom_lr, weight_scaling=weight_scaling)
        self.mean_layer = EqualisedLinearLayer(input_size, num_features, bias=True,
                                               custom_lr=custom_lr, weight_scaling=weight_scaling)

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

class ColoringLayer(nn.Module):
    def __init__(self, cfg, ic, ks, pad='reflect'):
        super().__init__()
        pad_size = int(np.ceil((ks - 1.0) / 2))
        self._pad = get_padding(pad)(pad_size)
        self._hidden_conv = EqualisedConvLayer(ic, cfg.ngf, ks, bias=True, weight_scaling=cfg.weight_scaling, custom_lr=cfg.color_custom_lr)
        self._hidden_act = nn.LeakyReLU(negative_slope=0.2)
        # might need to be removed, don't know
        # or replaced with StyleNorm
        self._hidden_norm = InstanceNorm2d(cfg.ngf)
        self._color_conv = EqualisedConvLayer(cfg.ngf, 3, ks, bias=False, weight_scaling=cfg.weight_scaling, custom_lr=cfg.color_custom_lr)
        self._color_act = nn.Tanh()

    def forward(self, input):
        x = self._pad(input)
        x = self._hidden_conv(x)
        x = self._hidden_act(x)
        x = self._hidden_norm(x)
        x = self._pad(x)
        x = self._color_conv(x)
        return self._color_act(x)

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

        _style_net = [EqualisedLinearLayer(self.input_dim, self._proj_dim, bias=cfg.style_bias, custom_lr=cfg.style_lr_mul,
                                           weight_scaling=cfg.weight_scaling),
                      nn.LeakyReLU(negative_slope=0.2),
                      nn.BatchNorm1d(self._proj_dim)]
        for _ in range(cfg.style_layers):
            _style_net += [EqualisedLinearLayer(self._proj_dim, self._proj_dim, bias=cfg.style_bias, custom_lr=cfg.style_lr_mul,
                                           weight_scaling=cfg.weight_scaling),
                            nn.LeakyReLU(negative_slope=0.2),
                            nn.BatchNorm1d(self._proj_dim)]
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
        self.pad =  get_padding(pad)(pad_size)

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


class StyleResBlock(nn.Module):
    def __init__(self, cfg, ic, oc, ks=3, pad='reflect', act='leaky_relu', upsample=False):
        super().__init__()
        pad_size = int(np.ceil((ks - 1.0) / 2))
        self.fwd_block = StyleBlock(cfg, ic, oc, ks, pad, act, upsample)

        if ic != oc:
            self.res_block = EqualisedConvLayer(ic, oc, ks, bias=True, weight_scaling=cfg.weight_scaling, custom_lr=1.)
            self.res_norm = AdaptiveStyleLayer(cfg.style_dim, oc, weight_scaling=cfg.weight_scaling, custom_lr=cfg.style_lr_mul)
            self.pad = get_padding(pad)(pad_size)
            self.act = get_activ(act)()
        else:
            self.res_block = None

    def forward(self, input, style):
        block_out, block_inp = self.fwd_block(input, style)
        res = block_inp
        if self.res_block:
            res = self.pad(block_inp)
            res = self.res_block(res)
            res = self.act(res)
            res = self.res_norm(style, res)
        return block_out + res

class GeneratorTrunk(nn.Module):
    def __init__(self, cfg, ic, oc, num_block):
        super().__init__()
        _layers = []
        if ic == oc:
            for _ in range(num_block):
                   _layers += [StyleResBlock(cfg, ic, oc)]
        else:
            _layers  += [StyleResBlock(cfg, ic, oc, upsample=True)]
            for _ in range(num_block - 1):
                _layers += [StyleResBlock(cfg, oc, oc, upsample=False)]
        self._layers = nn.ModuleList(_layers)

    def forward(self, x, style):
        for _layer in self._layers:
            x = _layer(x, style)
        return x

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
        generator_bodies += [GeneratorTrunk(cfg, cfg.ngf * 8, cfg.ngf * 4, num_block=2)]
        generator_bodies += [GeneratorTrunk(cfg, cfg.ngf * 4, cfg.ngf * 2, num_block=2)]
        generator_bodies += [GeneratorTrunk(cfg, cfg.ngf * 2, cfg.ngf * 1, num_block=2)]
        self.generator_bodies = nn.ModuleList(generator_bodies)
        self.generator_heads = nn.ModuleList([ColoringLayer(cfg, cfg.ngf * f, 3, pad='reflect') for f in [4, 2, 1]])

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
        # TODO: for style mixing, use fixed noise
        results = []
        for _layer, _head in zip(self.generator_bodies, self.generator_heads):
            feats = _layer(feats, style)
            results += [_head(feats)]

        return results