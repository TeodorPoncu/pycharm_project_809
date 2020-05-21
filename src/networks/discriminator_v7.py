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
        out = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return out


class EqualisedConvBlock(nn.Module):
    def __init__(self, cfg, ic, oc, ks, act='none', pad='none', norm='none'):
        super().__init__()
        pad_size = int(np.ceil((ks - 1.0) / 2))
        self.act = get_activ(act)()
        self.pad = get_padding(pad)(pad_size)
        self.norm = get_norm(norm)(oc)
        self.conv = EqualisedConvLayer(ic=ic, oc=oc, ks=ks, weight_scaling=cfg.weight_scaling)

    def forward(self, input):
        x = self.pad(input)
        x = self.conv(x)  # maybe should do the bias thingy
        x = self.act(x)
        x = self.norm(x)
        return x


class fRGB(nn.Module):
    def __init__(self, cfg, oc):
        super().__init__()
        self._transform = nn.Sequential(*[
            EqualisedConvLayer(3, oc, 1, weight_scaling=cfg.weight_scaling),
            nn.LeakyReLU(negative_slope=0.2)
        ])

    def forward(self, x):
        return self._transform(x)


class BlurConv(nn.Module):
    def __init__(self, f):
        super().__init__()
        kernel = [sqrt(f)] * f
        self.kernel = torch.tensor(kernel, dtype=torch.float32)
        self.kernel = self.kernel[:, None] * self.kernel[None, :]
        self.kernel = self.kernel[None, None]
        self.stride = 2

    def forward(self, input):
        kernel = self.kernel.expand(input.size(1), input.size(1), -1, -1).to(input.device)
        x = F.conv2d(input, kernel, stride=1, padding=int(np.ceil((self.kernel.size(2) - 1.0) / 2)))
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

    def __call__(self, input, target_is_real=True, for_discriminator=True):
        target_tensor = self.get_target_tensor(input, target_is_real, for_discriminator)
        loss = self.loss(input, target_tensor.to(input.device))
        return loss


class ResidualLayer(nn.Module):
    def __init__(self, cfg, ic, oc, ks=3, blur=None, blur_f=3):
        super().__init__()
        self.cfg = cfg
        self._layer = nn.Sequential(*[
            BlurConv(blur_f),
            EqualisedConvBlock(cfg, ic=ic, oc=oc, ks=ks, pad='zeros', act='leaky_relu', norm='batch'),
        ])

        if oc != ic:
            if blur is None:
                self._residual = nn.Sequential(*[
                    BlurConv(blur_f),
                    EqualisedConvLayer(ic, oc, ks=1, weight_scaling=cfg.weight_scaling)
                ])
            else:
                self._residual = nn.Sequential(*[
                    blur,
                    EqualisedConvLayer(ic, oc, ks=1, weight_scaling=cfg.weight_scaling)
                ])
        else:
            if blur is None:
                self._residual = BlurConv(blur_f)
            else:
                self._residual = None

    def forward(self, x):
        if self._residual is not None:
            out = self._layer(x) + self._residual(x)
        else:
            out = self._layer(x) + x
        out = out / (2 ** 0.5)
        return out


class DownsampleLayer(nn.Module):
    def __init__(self, cfg, ic, oc, ks=3, f=2):
        super().__init__()
        self.cfg = cfg
        pad_size = int(np.ceil((ks - 1.0) / 2))

        self._downsample = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=(f, f), stride=f),
            nn.ZeroPad2d(pad_size),
            EqualisedConvLayer(ic, oc, ks, weight_scaling=cfg.weight_scaling)
        ])

    def forward(self, x):
        return self._downsample(x)


class FeatureConvolutionalDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.load_size
        self._blur_layer = BlurConv(3)
        self._build_feature_layers(cfg)
        self._build_u_head(cfg)
        self._build_c_head(cfg)

    def _build_feature_layers(self, cfg):
        _layers = [fRGB(cfg, cfg.ndf), self._blur_layer]
        f = 1
        pf = 1

        while self.input_dim != 2:
            f = min(8, f * 2)
            _layers += [ResidualLayer(cfg, cfg.ndf * pf, cfg.ndf * pf, blur=self._blur_layer)]
            _layers += [DownsampleLayer(cfg, cfg.ndf * pf, cfg.ndf * f)]

            pf = f
            self.input_dim = int(self.input_dim / 2)

        self._body = nn.Sequential(*_layers)

    def _build_u_head(self, cfg):
        _layers = []
        _layers += [
            EqualisedLinearLayer(32 * cfg.ndf, 32 * cfg.ndf, weight_scaling=cfg.weight_scaling),
            nn.BatchNorm1d(32 * cfg.ndf),
            nn.LeakyReLU(negative_slope=0.2),
            EqualisedLinearLayer(32 * cfg.ndf, 1, weight_scaling=cfg.weight_scaling)

        ]
        self._u_head = nn.Sequential(*_layers)

    def _build_c_head(self, cfg):
        _layers = []
        _layers += [
            EqualisedLinearLayer(32 * cfg.ndf + cfg.style_dim, 32 * cfg.ndf + cfg.style_dim, weight_scaling=cfg.weight_scaling),
            nn.BatchNorm1d(32 * cfg.ndf + cfg.style_dim),
            nn.LeakyReLU(negative_slope=0.2),
            EqualisedLinearLayer(32 * cfg.ndf + cfg.style_dim, 1,  weight_scaling=cfg.weight_scaling)
        ]
        self._c_head = nn.Sequential(*_layers)

    def forward_u_head(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        u_predicts = self._u_head(x)
        return u_predicts

    def forward_c_head(self, x, c):
        feats = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        feats = torch.cat([feats, c], dim=1)
        c_predicts = self._c_head(feats)
        return c_predicts

    def forward(self, x, code=None):
        if code == None:
            score = self.forward_u_head(self._body(x))
        else:
            score = self.forward_c_head(self._body(x), code)
        return score


def discriminator_regularization(real_logits, real_input, fake_logits, fake_input):
    grad_real_logits = torch.autograd.grad(
        outputs=real_logits,
        inputs=real_input,
        grad_outputs=torch.ones_like(real_logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_fake_logits = torch.autograd.grad(
        outputs=fake_logits,
        inputs=fake_input,
        grad_outputs=torch.ones_like(fake_logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm_grad_real = grad_real_logits.view(grad_real_logits.size(0), -1).norm(2, dim=1) ** 2
    norm_grad_fake = grad_fake_logits.view(grad_real_logits.size(0), -1).norm(2, dim=1) ** 2

    real_pred = torch.sigmoid(real_logits)
    fake_pred = torch.sigmoid(fake_logits)

    reg_real = torch.square(1.0 - real_pred) * torch.square(norm_grad_real)
    reg_fake = torch.square(fake_pred) * torch.square(norm_grad_fake)
    reg_term = torch.mean(reg_real + reg_fake)
    return reg_term
