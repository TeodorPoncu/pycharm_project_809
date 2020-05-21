import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.basic import *
from ..layers.normalization import InstanceNorm2d

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.real_label = cfg.real_label - cfg.smooth
        self.fake_label = cfg.fake_label + cfg.smooth
        self.generator_label = cfg.real_label

        self.real_label_tensor = None
        self.fake_label_tensor = None
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
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def __call__(self, input, condition=None, condition_similarity=1, target_is_real=True, for_discriminator=True):
        target_tensor = self.get_target_tensor(input, target_is_real, for_discriminator)
        loss = F.binary_cross_entropy(input, target_tensor.to(input.device))
        return loss


class PatchConvolutionalDiscriminator(nn.Module):
    def __init__(self, input_dim, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.criterion = BinaryCrossEntropyLoss(cfg)
        self._build_feature_layers(input_dim, cfg)
        self._build_u_head(cfg)
        #self._build_c_head(cfg)

    def _build_feature_layers(self, input_dim, cfg):
        _layers = []
        _layers = _layers + [ConvBlock(ic=3, oc=cfg.ndf, ks=7, pad='reflect', act='leaky_relu', norm='batch', spectral=cfg.dsc_spectral)]
        _conv_dim = input_dim
        self.input_dim = input_dim
        self.init = input_dim
        f = 1
        pf = 1

        while self.input_dim != 8:
            f = min(8, f * 2)
            _layers = _layers + [ConvBlock(ic=cfg.ndf * pf, oc=cfg.ndf * f, ks=3, s=2, pad='reflect', act='leaky_relu', norm='batch',
                                           spectral=cfg.dsc_spectral)]
            pf = f
            self.input_dim = int(self.input_dim/2)

        self.feature_extractor = nn.Sequential(*_layers)
        self._factor = f

    def _build_u_head(self, cfg):
        #_u_head = [nn.GLU(dim=1)]
        _u_head=  []
        _u_head = _u_head + [ConvBlock(ic=cfg.ndf * self._factor, oc=cfg.ndf * self._factor, ks=3, pad='reflect',
                             act='leaky_relu', norm='batch', spectral=cfg.dsc_spectral)]
        _u_head = _u_head + [ConvBlock(ic=cfg.ndf * self._factor, oc=1, ks=1, pad='none',
                                       act='none', norm='none', spectral=cfg.dsc_spectral)]
        #self._u_proj = ConvBlock(ic=cfg.ndf * self._factor, oc=cfg.ndf * self._factor, ks=1, pad='none',
        #                         act='leaky_relu', norm='batch', spectral=cfg.dsc_spectral)
        self._u_head = nn.Sequential(*_u_head)

    def forward_u_head(self, x):
        #u_sig_features = self._u_proj(x)
        #u_features = torch.cat([x, u_sig_features], dim=1)
        u_predicts = self._u_head(x)
        return u_predicts

    def forward_features(self, x):
        features = self.feature_extractor(x)
        return features

    def forward(self, x, aggregated_for_discriminator=True, code=None):
        if aggregated_for_discriminator:
            real_images = x[:self.cfg.pool_real_size]
            fake_images = x[self.cfg.pool_real_size:]
            real_feats = self.forward_features(real_images)
            fake_feats = self.forward_features(fake_images)

            if code==None:
                real_score = self.forward_u_head(real_feats)
                fake_score = self.forward_u_head(fake_feats)
            return real_score, fake_score
        else:
            feats =  self.forward_features(x)
            score = self.forward_u_head(feats)
            return score

    def compute_loss(self, scores, conditioned, is_real=True):
        if conditioned:
            loss_real = self.criterion_d(scores[0], target_is_real=True)
            loss_fake = self.criterion_d(scores[1], target_is_real=False)
        else:
            loss = self.criterion(scores, target_is_real=is_real, for_discriminator=True)
        return loss


    def forward_c_head(self, features, c):
        c = c.view(c.size(0), -1, 8, 8)
        c = c.repeat(features.size(0), 1, 1, 1)
        c_features = self._c_up_head(c)
        c_features = torch.cat([features, c_features], dim=1)
        c_predicts = self._c_head(c_features)
        return c_predicts
