from ..layers import get_activ, get_padding
import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self, input_dim, device, cfg):
        super().__init__()
        self.device = device
        self._linear_mu = nn.Linear(input_dim, cfg.latent_dim)
        self._linear_ln = nn.Linear(input_dim, cfg.latent_dim)
        self._act = get_activ(cfg.vae_act)
        if self._act is not None:
            self._act = self._act(inplace=False)

    def extract_param(self, x):
        mu = self._linear_mu(x)
        log_var = self._linear_ln(x)
        if self._act is not None:
            mu = self._act(mu)
            log_var = self._act(log_var)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, log_var = self.extract_param(x)
        code = self.reparametrize(mu, log_var)
        return mu, log_var, code


class GLUVAE(nn.Module):
    def __init__(self, input_dim, device, cfg):
        super().__init__()
        self.device = device
        if cfg.add_noise:
            self.latent_dim = int(cfg.latent_dim / 2)
            self.noise = True
        else:
            self.latent_dim = cfg.latent_dim
            self.noise = False

        self._proj_mu = nn.Linear(input_dim, self.latent_dim)
        self._proj_ln = nn.Linear(input_dim, self.latent_dim)
        self._act = nn.GELU()

    def extract_param(self, x):
        mu = self._proj_mu(x)
        log_var = self._proj_ln(x)
        mu = self._act(mu)
        log_var = self._act(log_var)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, log_var = self.extract_param(x)
        code = self.reparametrize(mu, log_var)
        return mu, log_var, code