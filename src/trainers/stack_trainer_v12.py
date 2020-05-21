from ..networks.generators_v10 import *
from ..networks.language_models import *
from ..networks.vae import *
from ..networks.base_model import *
from ..networks.discriminator_v8 import FeatureConvolutionalDiscriminator, BinaryCrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.rmsprop import RMSprop
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torchvision.utils as vutils
from random import randint, uniform, sample
from transformers import pipeline
from transformers import RobertaTokenizerFast
import torch
import dataset
import transformers
from transformers import LineByLineTextDataset
from transformers import RobertaConfig, RobertaTokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import pipeline
from qhoptim.pyt import QHAdam
from torch.optim import Adam


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
    def __init__(self, cfg, device):
        super().__init__()
        tokenizer = RobertaTokenizerFast.from_pretrained('./bird_bpe_vocab', max_len=256)
        _config = RobertaConfig(
            vocab_size=tokenizer._tokenizer.get_vocab_size(),
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            max_position_embeddings=256,
            pad_token_id=1,
            eos_token_id=0,
            bos_token_id=2,
            output_attentions=False,
            output_hidden_states=False
        )
        _model = RobertaForMaskedLM(_config)
        _model.load_state_dict(torch.load('bert_small/checkpoint-1100/pytorch_model.bin'))
        _model.eval()
        self.tokenizer = tokenizer
        self._model = _model
        self.device = device
        self.pad_token = 0
        self.batch_size = cfg.batch_size
        self.proj = None
        if cfg.proj_lang:
            self.proj = nn.Sequential(*[nn.Linear(512, cfg.latent_dim), nn.Tanh()])

    def forward(self, input):
        if not isinstance(input, list):
            input = [input]
        input = self.tokenizer.batch_encode_plus(input, pad_to_max_length=False, return_tensors='pt')
        emb_out, pool_out = self._model.roberta(input_ids=input['input_ids'].to(self.device),
                                                attention_mask=input['attention_mask'].to(self.device))
        if self.proj is not None:
            return self.proj(emb_out.mean(dim=1))
        else:
            return emb_out.mean(dim=1)


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

        self.language_model = LanguageModel(cfg, self.device_map['style']).to(self.device_map['style'])
        self.content_model = VAE(cfg.rnn_hidden_dim, self.device_map['style'], cfg).to(self.device_map['style'])
        self.style_model = VAE(cfg.rnn_hidden_dim, self.device_map['style'], cfg).to(self.device_map['style'])

        self.generator = StyleGenerator(cfg)
        self.discriminators = {}
        for scale in self.scales:
            self.discriminators[scale] = FeatureConvolutionalDiscriminator(cfg, scale)
        self.discriminators = nn.ModuleDict(self.discriminators)
        self.visual_names = ['visual_dict']
        self.visual_dict = {'real': None, 'fake': None}
        self.loss_names = ['loss']
        self.visualizer = Visualizer(cfg)

        self.consistency_criterion = ColorConsistencyLoss(cfg).to(self.device_map['img'])
        self.distribution_criterion = KLDLoss().to(self.device_map['img'])
        self.generator_criterion = BinaryCrossEntropyLoss(cfg).to(self.device_map['img'])

        #self.reset_parameters()
        self.generator = self.generator.to(self.device_map['img'])
        self.discriminators = self.discriminators.to(self.device_map['img'])

        self.latent_scale = int(cfg.load_size // (2 ** 6))
        self.latent_channels = int(cfg.latent_dim) // (self.latent_scale ** 2)
        self.channels_z = 16 * self.cfg.ngf - self.latent_channels

    def reset_parameters(self):
        self.init_weights(self.generator, self.cfg.gen_init, self.cfg.gen_gain)
        self.init_weights(self.style_model, self.cfg.lmf_init, self.cfg.lmf_gain)
        self.init_weights(self.content_model, self.cfg.lmf_init, self.cfg.lmf_gain)
        self.init_weights(self.discriminators, self.cfg.dsc_init, self.cfg.dsc_gain)
        if self.cfg.proj_lang:
            self.init_weights(self.language_model.proj, self.cfg.lmf_init, self.cfg.lmf_gain)

    def set_input(self, input):
        self.real_images = {}
        text_input = input['text']
        choices = [tup for tup in text_input[0]]
        self.text_input = choices

        miss_input = input['miss']
        choices = [tup for tup in miss_input[0]]
        self.miss_input = choices

        self.visual_dict['real'] = input['images']
        for scale in input['images']:
            self.real_images[scale] = input['images'][scale].to(self.devices[0])

    def forward_discriminator(self, input, code=None):
        pred = {}
        for scale in self.scales:
            pred[scale] = self.discriminators[scale](input[scale], code)
        return pred

    def backward_discriminator(self):
        self.loss_real_c = 0.
        self.loss_real_u = 0.
        self.loss_fake_c = 0.
        self.loss_fake_u = 0.
        self.loss_real_m = 0.
        for scale in self.scales:
            scores_real_u = self.discriminators[scale](self.real_images[scale])
            scores_real_c = self.discriminators[scale](self.real_images[scale], code=self.style_feat)
            scores_fake_u = self.discriminators[scale](self.fake_images[scale].detach())
            scores_fake_c = self.discriminators[scale](self.fake_images[scale].detach(), code=self.style_feat)
            scores_real_m = self.discriminators[scale](self.real_images[scale], code=self.miss_style_feat)
            self.loss_real_u += self.generator_criterion(scores_real_u, target_is_real=True, for_discriminator=True)
            self.loss_real_c += self.generator_criterion(scores_real_c, target_is_real=True, for_discriminator=True)
            self.loss_real_m += self.generator_criterion(scores_real_m, target_is_real=False, for_discriminator=True)
            self.loss_fake_u += self.generator_criterion(scores_fake_u, target_is_real=False, for_discriminator=True)
            self.loss_fake_c += self.generator_criterion(scores_fake_c, target_is_real=False, for_discriminator=True)
        loss = self.loss_real_u * self.cfg.lambda_uncond + \
               self.loss_real_c * self.cfg.lambda_match + \
               self.loss_real_m * self.cfg.lambda_match + \
               self.loss_fake_u * self.cfg.lambda_uncond + \
               self.loss_fake_c * self.cfg.lambda_match
        self.loss_kld_style = self.distribution_criterion(self.style_mu, self.style_log_var) * self.cfg.lambda_kld + \
                              self.distribution_criterion(self.miss_style_mu,
                                                          self.miss_style_log_var) * self.cfg.lambda_kld
        loss *= 0.2
        loss += self.loss_kld_style * 0.5
        loss.backward()

    def forward_generator(self):
        text_embeddings = self.language_model(self.text_input)
        self.style_mu, self.style_log_var, self.style_feat = self.style_model(text_embeddings)
        self.content_mu, self.content_log_var, self.content_feat = self.content_model(text_embeddings.detach())

        miss_embeddings = self.language_model(self.miss_input)
        self.miss_style_mu, self.miss_style_log_var, self.miss_style_feat = self.style_model(miss_embeddings)
        self.miss_content_mu, self.miss_content_log_var, self.miss_content_feat = self.content_model(miss_embeddings.detach())

        interp_factor = torch.rand(1).to(self.device_map['img'])
        interp_factor = torch.clamp(interp_factor, 0.01, 0.95)
        self.interp_content = (1 - interp_factor) * self.content_feat + interp_factor * self.miss_content_feat

        images = self.generator(self.style_feat.to(self.device_map['img']).detach(), self.content_feat.to(self.device_map['img']))
        inter_s = self.generator(self.style_feat.to(self.device_map['img']).detach(), self.interp_content.to(self.device_map['img']))

        self.fake_images = {}
        self.interp_images = {}

        for idx, scale in enumerate(self.scales):
            self.fake_images[scale] = images[idx]
            self.interp_images[scale] = inter_s[idx]

    def backward_generator(self):
        loss = 0.
        self.loss_gen_u = 0.
        self.loss_gen_c = 0.
        self.loss_gen_i = 0.

        for scale in self.scales:
            pred_u = self.discriminators[scale](self.fake_images[scale])
            pred_c = self.discriminators[scale](self.fake_images[scale], code=self.style_feat.detach())
            pred_i = self.discriminators[scale](self.interp_images[scale])
            self.loss_gen_u += self.generator_criterion(pred_u, target_is_real=True, for_discriminator=False)
            self.loss_gen_c += self.generator_criterion(pred_c, target_is_real=True, for_discriminator=False)
            self.loss_gen_i += self.generator_criterion(pred_i, target_is_real=True, for_discriminator=False)
            loss += self.loss_gen_u * self.cfg.lambda_uncond + \
                    self.loss_gen_c * self.cfg.lambda_match + \
                    self.loss_gen_i * self.cfg.lambda_interp
        loss /= 9
        self.loss_consistency = self.consistency_criterion(list(self.fake_images.values())) * self.cfg.lambda_clr + \
                                self.consistency_criterion(list(self.interp_images.values())) * self.cfg.lambda_clr
        self.loss_kld_content = self.distribution_criterion(self.content_mu, self.content_log_var) * self.cfg.lambda_kld + \
                                self.distribution_criterion(self.miss_content_mu, self.miss_content_log_var) * self.cfg.lambda_kld
        loss += self.loss_consistency * 0.5 + self.loss_kld_content * 0.5
        loss.backward()

    def optimize(self):
        self.optim_generator.zero_grad()
        self.optim_content.zero_grad()
        self.set_requires_grad([self.discriminators], requires_grad=False)
        self.forward_generator()
        self.backward_generator()
        #nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=100.)
        #nn.utils.clip_grad_norm_(self.content_model.parameters(), max_norm=100.)
        self.optim_generator.step()
        self.optim_content.step()

        self.set_requires_grad([self.discriminators], requires_grad=True)
        for _ in range(self.cfg.critic_iter):
            self.optim_style.zero_grad()
            self.optim_discriminator.zero_grad()
            self.backward_discriminator()
            # nn.utils.clip_grad_norm_(self.discriminators.parameters(), max_norm=100.)
            # nn.utils.clip_grad_norm_(self.style_model.parameters(), max_norm=100.)
            # nn.utils.clip_grad_norm_(self.language_model.parameters(), max_norm=1000.)
            self.optim_style.step()
            self.optim_discriminator.step()
            if self.cfg.proj_lang:
                self.optim_language.step()

        self.loss_loss = {'real_u': self.loss_real_u,
                          'real_c': self.loss_real_c,
                          'real_m': self.loss_real_m,
                          'fake_u': self.loss_fake_u,
                          'fake_c': self.loss_fake_c,
                          'gen_u': self.loss_gen_u,
                          'gen_c': self.loss_gen_c,
                          'gen_i': self.loss_gen_i,
                          'consistency': self.loss_consistency,
                          'kld_style': self.loss_kld_style,
                          'kld_content': self.loss_kld_content}

        self.visual_dict['fake'] = self.fake_images

    # build dynamically sized layers, then resets all parameters
    def init_trainer_network(self):
        nus = (0.7, 1.0)
        self.gen_paramas = list(self.generator.parameters())
        self.dsc_params = list(self.discriminators.parameters())
        if self.cfg.proj_lang:
            self.language_params = list(self.language_model.proj.parameters())
            self.optim_language = QHAdam(self.language_params, lr=self.cfg.lang_lr, betas=self.cfg.lang_betas, nus=(0.7, 0.8))
        self.style_params = list(self.style_model.parameters())
        self.content_params = list(self.content_model.parameters())
        self.optim_generator = QHAdam(self.gen_paramas, lr=self.cfg.gen_lr, betas=self.cfg.gen_betas, nus=(0.7, 0.8))
        self.optim_discriminator = QHAdam(self.dsc_params, lr=self.cfg.dsc_lr, betas=self.cfg.dsc_betas, nus=(0.7, 0.8))
        self.optim_style = QHAdam(self.style_params, lr=self.cfg.lmf_lr, betas=self.cfg.lmf_betas, nus=(0.7, 0.8))
        self.optim_content = QHAdam(self.content_params, lr=self.cfg.lmf_lr, betas=self.cfg.lmf_betas, nus=(0.7, 0.8))

    def get_current_visuals(self):
        dicts = super().get_current_visuals()
        image = self.visualizer.write_visuals(dicts)
        return image
