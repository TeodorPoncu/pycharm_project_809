from ..networks.generators_v6 import *
from ..networks.language_models import *
from ..networks.vae import *
from ..networks.base_model import *
from ..networks.discriminator_v4 import PatchConvolutionalDiscriminator, BinaryCrossEntropyLoss
from torch.optim.adamw import AdamW
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torchvision.utils as vutils
from random import randint, uniform, choice, sample

class Visualizer():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.device = 'cuda:0' + str(device)

    def write_visuals(self, visuals_dict):
        def resize(img):
            return F.interpolate(img, size=(self.cfg.load_size, self.cfg.load_size), mode='nearest')

        keys = list(visuals_dict.keys())
        real = visuals_dict[keys[0]]['real']
        fake = visuals_dict[keys[0]]['fake']

        imgs_real = torch.ones(3, 3, self.cfg.load_size, self.cfg.load_size)
        imgs_fake = torch.ones(3, 3, self.cfg.load_size, self.cfg.load_size)

        self.scales = [str((self.cfg.load_size // (2 ** i))) for i in range(3)]

        for idx, scale in enumerate(self.scales):
            imgs_real[idx] = resize(real[scale].detach().cpu()[0].unsqueeze(0))
            imgs_fake[idx] = resize(fake[scale].detach().cpu()[0].unsqueeze(0))

        imgs = torch.cat([imgs_real, imgs_fake], dim=0)
        image_grid = vutils.make_grid(imgs, nrow=3, padding=2, normalize=True)
        return image_grid

class ImagePool():
    def __init__(self, opt):
        self.real_images = []
        self.fake_images = []
        self.max_real_size = opt.pool_real_size
        self.max_fake_size = opt.pool_fake_size
        self.add_prob = opt.pool_prob
        self.real_size = 0
        self.fake_size = 0

    def update(self, real_images, fake_images):
        for image in real_images:
            image = torch.unsqueeze(image.data, 0)
            while self.real_size < self.max_real_size:
                self.real_size = self.real_size + 1
                self.real_images = self.real_images + [image]
            else:
                # dump earliest image
                self.real_images = [image] + self.real_images[1:]
        for image in fake_images:
            image = torch.unsqueeze(image.data, 0).detach()
            while self.fake_size < self.max_fake_size:
                self.fake_size = self.fake_size + 1
                self.fake_images = self.fake_images + [image]
            else:
                p = uniform(0, 1)
                if p > 1 - self.add_prob:
                    swap_idx = randint(0, len(self.fake_images) - 1)
                    self.fake_images[swap_idx] = image

    def fetch_real(self):
        return torch.cat(self.real_images, dim=0)

    def fetch_fake(self):
        return torch.cat(self.fake_images, dim=0)


class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class ColorConsistencyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambda_cmean = cfg.lambda_cmean
        self.lambda_cvar = cfg.lambda_cvar

    def __call__(self, targets):
        loss = 0.
        for idx in range(len(targets) - 1):
            mean_1, covar_1 = self.get_mean_covar(targets[idx])
            mean_2, covar_2 = self.get_mean_covar(targets[idx + 1])
            loss += torch.norm(mean_1 - mean_2, p=2).pow(2) * self.lambda_cmean + \
                    torch.norm(covar_1 - covar_2, p='fro').pow(2) * self.lambda_cvar
        return loss

    def get_mean_covar(self, img):
        mean = img.mean(dim=(2, 3), keepdim=True)
        var = (img - mean)
        var_t = img.permute(0, 1, 3, 2)
        covar = var @ var_t
        covar = covar.sum(dim=(2, 3)) / (img.size(2) * img.size(3))
        return mean, covar

class LanguageModel(nn.Module):
    def __init__(self, cfg, tokenizer, device):
        super().__init__()
        self._model = RNN(cfg, tokenizer.get_vocab_size())
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input):
        if not isinstance(input, list):
            input = [input]
        ids = [self.tokenizer.encode(string[0]).ids for string in input]
        ids = torch.tensor(ids).to(self.device)
        out = self._model(ids)
        return out

class StackTrainer(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scales = [str((cfg.load_size // (2 ** i))) for i in range(3)]
        self.scales.reverse()

        self.device_map = {'style': self.devices[0],
                           'content': self.devices[0],
                           'img': self.devices[0]}
        self.network_names = ['style_model',
                              'content_model',
                              'generator',
                              'discriminators']
        self.device_name_map = {'style_model': 'style',
                                'content_model': 'content',
                                'generators': 'img',
                                'discriminators': 'img'}

        tokenizer = ByteLevelBPETokenizer(
            "vocab.json",
            "merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )

        self.style_l_model = LanguageModel(cfg, tokenizer, self.device_map['style']).to(self.device_map['style'])
        self.style_f_model = VAE(cfg.rnn_hidden_dim * 2, self.device_map['style'], cfg).to(self.device_map['style'])

        self.content_l_model = LanguageModel(cfg, tokenizer, self.device_map['content']).to( self.device_map['content'])
        self.content_f_model = VAE(cfg.rnn_hidden_dim * 2, self.device_map['content'], cfg).to( self.device_map['content'])

        self.generator = StyleGenerator(cfg).to(self.device_map['img'])
        self.discriminators = {}
        for scale in self.scales:
            self.discriminators[scale] = PatchConvolutionalDiscriminator(int(scale), self.device_map['img'], cfg)
        self.discriminators = nn.ModuleDict(self.discriminators).to(self.device_map['img'])

        self.visual_names = ['visual_dict']
        self.visual_dict = {'real': None, 'fake': None}
        self.loss_names = ['loss']
        self.visualizer = Visualizer(cfg)
        self.image_pool = {scale: ImagePool(cfg) for scale in self.scales}

        self.consistency_criterion = ColorConsistencyLoss(cfg).to(self.device_map['img'])
        self.distribution_criterion = KLDLoss().to(self.device_map['img'])
        self.generator_criterion = BinaryCrossEntropyLoss(cfg).to(self.device_map['img'])

    def reset_parameters(self):
        self.init_weights(self.generator, self.cfg.gen_init, self.cfg.gen_gain)
        self.init_weights(self.style_f_model, self.cfg.lmf_init, self.cfg.lmf_gain)
        self.init_weights(self.style_l_model, self.cfg.lmf_init, self.cfg.lmf_gain)
        self.init_weights(self.content_f_model, self.cfg.lmf_init, self.cfg.lmf_gain)
        self.init_weights(self.content_l_model, self.cfg.lmf_init, self.cfg.lmf_gain)
        self.init_weights(self.discriminators, self.cfg.dsc_init, self.cfg.dsc_gain)

    def set_input(self, input):
        self.real_images = {}
        text_input = input['text']
        choices = sample(text_input, self.cfg.batch_size)
        self.text_input = [choices[0] for choice in choices]
        self.visual_dict['real'] = input['images']
        for scale in input['images']:
            self.real_images[scale] = input['images'][scale].to(self.devices[0])

    def forward_discriminator(self, input, code=None):
        pred = {}
        for scale in self.scales:
            pred[scale] = self.discriminators[scale](input[scale], code)
        return pred

    def backward_discriminator(self):
        loss = 0.
        self.loss_real_d = 0.
        self.loss_fake_d = 0.
        for scale in self.scales:
            real_images = self.image_pool[scale].fetch_real()
            fake_images = self.image_pool[scale].fetch_fake()
            scores_real = self.discriminators[scale](real_images, aggregated_for_discriminator=False)
            scores_fake = self.discriminators[scale](fake_images, aggregated_for_discriminator=False)
            self.loss_real_d += self.discriminators[scale].compute_loss(scores_real, conditioned=False, is_real=True,)
            self.loss_fake_d += self.discriminators[scale].compute_loss(scores_fake, conditioned=False, is_real=False)
            loss += self.loss_real_d + self.loss_fake_d
        loss *= 0.5
        loss.backward()

    def forward_generator(self):
        style_embeddings = self.style_l_model(self.text_input)
        self.style_mu, self.style_log_var, style_feat = self.style_f_model(style_embeddings)

        content_embeddings = self.content_l_model(self.text_input)
        self.content_mu, self.content_log_var, content_feat = self.style_f_model(content_embeddings)

        images = self.generator(style_feat.to(self.device_map['img']), content_feat.to(self.device_map['img']))
        results = {}

        for idx, scale in enumerate(self.scales):
            results[scale] = images[idx]
            self.image_pool[scale].update(real_images=self.real_images[scale], fake_images=results[scale].detach())

        return results

    def backward_generator(self, images):
        loss = 0.
        self.loss_gen = {}
        for scale in self.scales:
            pred = self.discriminators[scale](images[scale], aggregated_for_discriminator=False)
            self.loss_gen[scale] = self.generator_criterion(pred, target_is_real=True, for_discriminator=False)
            loss += self.loss_gen[scale]
        self.loss_consistency = self.consistency_criterion(list(images.values())) * self.cfg.lambda_clr
        self.loss_kld_style = self.distribution_criterion(self.style_mu, self.style_log_var) * self.cfg.lambda_kld
        self.loss_kld_content = self.distribution_criterion(self.content_mu, self.content_log_var) * self.cfg.lambda_kld
        loss += self.loss_consistency + self.loss_kld_style + self.loss_kld_content
        loss.backward()

    def optimize(self):
        self.optim_generator.zero_grad()
        self.optim_style.zero_grad()
        self.optim_content.zero_grad()
        self.set_requires_grad([self.discriminators], requires_grad=False)
        self.fake_images = self.forward_generator()
        self.backward_generator(self.fake_images)
        self.optim_generator.step()
        self.optim_style.step()
        self.optim_content.step()

        self.set_requires_grad([self.discriminators], requires_grad=True)
        for _ in range(self.cfg.critic_iter):
            self.optim_discriminator.zero_grad()
            self.backward_discriminator()
            self.optim_discriminator.step()

        self.loss_loss = {'dsc_real': self.loss_real_d,
                          'dsc_fake': self.loss_fake_d,
                          'gen': self.loss_gen,
                          'consistency': self.loss_consistency,
                          'kld_style': self.loss_kld_style,
                          'kld_content': self.loss_kld_content}

        self.visual_dict['fake'] = self.fake_images

    # build dynamically sized layers, then resets all parameters
    def init_trainer_network(self):
        self.reset_parameters()

        gen_paramas = list(self.generator.parameters())
        dsc_params = list(self.discriminators.parameters())
        style_params = list(self.style_f_model.parameters()) + list(self.style_l_model.parameters())
        content_params = list(self.content_f_model.parameters()) + list(self.content_l_model.parameters())
        self.optim_generator = AdamW(gen_paramas, lr=self.cfg.gen_lr, betas=self.cfg.gen_betas, weight_decay=self.cfg.gen_wd)
        self.optim_discriminator = AdamW(dsc_params, lr=self.cfg.dsc_lr, betas=self.cfg.dsc_betas, weight_decay=self.cfg.dsc_wd)
        self.optim_style = AdamW(style_params, lr=self.cfg.lmf_lr, betas=self.cfg.lmf_betas, weight_decay=self.cfg.lmf_wd)
        self.optim_content = AdamW(content_params, lr=self.cfg.lmf_lr, betas=self.cfg.lmf_betas, weight_decay=self.cfg.lmf_wd)

    def get_current_visuals(self):
        dicts = super().get_current_visuals()
        image = self.visualizer.write_visuals(dicts)
        return image
