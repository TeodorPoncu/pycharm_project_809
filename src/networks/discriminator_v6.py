import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.basic import *
from ..layers.normalization import InstanceNorm2d

class ViewLayer(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class EqualisedConvBlock(nn.Module):
    def __init__(self, cfg, ic, oc, ks, act='none', pad='none', norm='none'):
        super().__init__()
        pad_size = int(np.ceil((ks - 1.0) / 2))
        self.act = get_activ(act)()
        self.pad = get_padding(pad)(pad_size)
        self.norm = get_norm(norm)(oc)
        self.conv = EqualisedConvLayer(ic=ic, oc=oc, ks=ks)

    def forward(self, input):
        x = self.pad(input)
        x = self.conv(x) # maybe should do the bias thingy
        x = self.act(x)
        x = self.norm(x)
        return x

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.real_label = cfg.real_label - cfg.smooth
        self.fake_label = cfg.fake_label
        self.generator_label = cfg.real_label
        self.loss = nn.BCEWithLogitsLoss()
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.code_tensor = None
        self.generator_label_tensor = None

    def get_target_tensor(self, input, target_is_real, for_discriminator):
        if for_discriminator:
            if target_is_real:
                if self.real_label_tensor is None:
                    self.real_label_tensor = torch.FloatTensor(1).fill_(self.real_label)
                    self.real_label_tensor.requires_grad_(False)
                return self.real_label_tensor.expand_as(input)
            else:
                if self.fake_label_tensor is None:
                    self.fake_label_tensor = torch.FloatTensor(1).fill_(self.fake_label)
                    self.fake_label_tensor.requires_grad_(False)
                return self.fake_label_tensor.expand_as(input)
        else:
            if self.generator_label_tensor is None:
                self.generator_label_tensor = torch.FloatTensor(1).fill_(self.generator_label)
                self.generator_label_tensor.requires_grad_(False)
            return self.generator_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def get_code_tensor(self, input, code):
        if self.code_tensor is None:
            self.code_tensor = torch.FloatTensor(1).fill_(1)
            self.code_tensor.requires_grad_(False)
        self.code_tensor = self.code_tensor * code
        self.code_tensor.requires_grad_(False)
        return self.code_tensor.expand_as(input)

    def __call__(self, input, condition=None, target_is_real=True, for_discriminator=True):
        if condition is not None:
            if not isinstance(condition, torch.Tensor):
                target_tensor = self.get_code_tensor(input, code=condition)
                loss = self.loss(input, target_tensor.to(input.device))
            else:
                loss = self.loss(input, condition.view(input.size(0), -1))
        else:
            target_tensor = self.get_target_tensor(input, target_is_real, for_discriminator)
            loss = self.loss(input, target_tensor.to(input.device))
        return loss


class FeatureConvolutionalDiscriminator(nn.Module):
    def __init__(self, input_dim, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.criterion = BinaryCrossEntropyLoss(cfg)
        self._build_feature_layers(input_dim, cfg)
        self._build_u_head(cfg)
        self._build_c_head(cfg)

    def _build_feature_layers(self, input_dim, cfg):
        _layers = []
        _layers = _layers + [EqualisedConvBlock(cfg, ic=3, oc=cfg.ndf, ks=7, pad='reflect', act='leaky_relu', norm='batch')]
        _conv_dim = input_dim
        self.input_dim = input_dim
        self.init = input_dim
        f = 1
        pf = 1

        while self.input_dim != 8:
            f = min(8, f * 2)
            _layers = _layers + [nn.AvgPool2d(kernel_size=(2, 2), stride=2)]
            _layers = _layers + [EqualisedConvBlock(cfg, ic=cfg.ndf * pf, oc=cfg.ndf * f, ks=3,
                                                    pad='reflect', act='leaky_relu', norm='batch')]
            pf = f
            self.input_dim = int(self.input_dim/2)

        self.feature_extractor = nn.Sequential(*_layers)
        self._factor = f

    def _build_u_head(self, cfg):
        _u_head=  []
        _u_head = _u_head + [EqualisedConvBlock(cfg, ic=cfg.ndf * self._factor, oc=cfg.ndf * self._factor, ks=3,
                                                pad='reflect', act='leaky_relu', norm='batch')]
        _u_head += [nn.AvgPool2d(kernel_size=(2, 2), stride=2)]
        _u_head = _u_head + [ViewLayer(-1)]
        _u_head = _u_head + [EqualisedLinearLayer(16 * cfg.ndf * self._factor, cfg.ndf * self._factor, bias=True),
                             nn.LeakyReLU(negative_slope=0.2),
                             nn.BatchNorm1d(num_features=cfg.ndf * self._factor)]
        _u_head = _u_head + [ViewLayer(-1)]
        _u_head = _u_head + [EqualisedLinearLayer(cfg.ndf * self._factor, 1, bias=True)]
        self._u_head = nn.Sequential(*_u_head)

    def _build_c_head(self, cfg):
        self._latent_channels = int(cfg.latent_dim / 64)
        _c_head = []
        _c_head = _c_head + [EqualisedConvBlock(cfg, ic=cfg.ndf * self._factor + self._latent_channels, oc=cfg.ndf * self._factor, ks=3,
                                                pad='reflect', act='leaky_relu', norm='batch')]
        _c_head += [nn.AvgPool2d(kernel_size=(2, 2), stride=2)]
        _c_head = _c_head + [
            EqualisedConvBlock(cfg, ic=cfg.ndf * self._factor, oc=cfg.ndf * self._factor, ks=3,
                               pad='reflect', act='leaky_relu', norm='batch')]
        _c_head = _c_head + [ViewLayer(-1)]
        _c_head = _c_head + [EqualisedLinearLayer(16 * cfg.ndf * self._factor, cfg.style_dim, bias=True),
                             nn.LeakyReLU(negative_slope=0.2),
                             nn.BatchNorm1d(num_features=cfg.style_dim)]

        _c_head = _c_head + [ViewLayer(-1)]
        _c_head = _c_head + [nn.Linear(cfg.style_dim, 1)]
        self._c_head = nn.Sequential(*_c_head)

    def forward_u_head(self, x):
        u_predicts = self._u_head(x)
        return u_predicts

    def forward_c_head(self, x, c):
        c = c.view(x.size(0), self._latent_channels, 8, 8)
        c = torch.cat([x, c], dim=1)
        c_predicts = self._c_head(c)
        return c_predicts

    def forward_features(self, x):
        features = self.feature_extractor(x)
        return features

    def forward(self, x, code=None):
        feats = self.forward_features(x)
        if code == None:
            score = self.forward_u_head(feats)
        else:
            score = self.forward_c_head(feats, code)
        return score
