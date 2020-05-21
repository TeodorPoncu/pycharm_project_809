from ..networks.generators_v9 import *
from ..networks.language_models import *
from ..networks.vae import *
from ..networks.base_model import *
from ..networks.discriminator_v7 import FeatureConvolutionalDiscriminator, BinaryCrossEntropyLoss, discriminator_regularization
from torch.optim.adamw import AdamW
from ..layers.basic import EqualisedLinearLayer
from torch.optim.rmsprop import RMSprop
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import transformers
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
            self.proj = nn.Sequential(*[EqualisedLinearLayer(512, cfg.latent_dim, weight_scaling=cfg.weight_scaling), nn.Tanh()])

    def forward(self, input):
        if not isinstance(input, list):
            input = [input]
        input = self.tokenizer.batch_encode_plus(input, pad_to_max_length=True, return_tensors='pt')
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

        self.cold = True

        self.language_model = LanguageModel(cfg, self.device_map['style']).to(self.device_map['style'])
        self.content_model = VAE(cfg.rnn_hidden_dim, self.device_map['style'], cfg).to(self.device_map['style'])
        self.style_model = VAE(cfg.rnn_hidden_dim, self.device_map['style'], cfg).to(self.device_map['style'])

        self.generator = StyleGenerator(cfg).to(self.device_map['img'])
        self.discriminator = FeatureConvolutionalDiscriminator(cfg).to(self.device_map['img'])

        self.visual_names = ['visual_dict']
        self.visual_dict = {'real': None, 'fake': None}
        self.loss_names = ['loss']
        self.visualizer = Visualizer(cfg)

        self.generator_criterion = BinaryCrossEntropyLoss(cfg).to(self.device_map['img'])
        self.consistency_criterion = ColorConsistencyLoss(cfg).to(self.device_map['img'])
        self.distribution_criterion = KLDLoss().to(self.device_map['img'])

        self.latent_scale = int(cfg.load_size // (2 ** 6))
        self.latent_channels = int(cfg.latent_dim) // (self.latent_scale ** 2)
        self.channels_z = 8 * self.cfg.ngf - self.latent_channels

    def reset_parameters(self):
        self.init_weights(self.style_model, self.cfg.lmf_init, self.cfg.lmf_gain)
        self.init_weights(self.content_model, self.cfg.lmf_init, self.cfg.lmf_gain)

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

    def forward_generator(self):
        self.language_embedding = self.language_model(self.text_input)
        self.style_mu, self.style_log_var, self.style_feat = self.style_model(self.language_embedding)
        self.content_mu, self.content_log_var, self.content_feat = self.content_model(self.language_embedding.detach())

        self.missmatch_embedding = self.language_model(self.miss_input)
        self.miss_style_mu, self.miss_style_log_var, self.miss_style_feat = self.style_model(self.missmatch_embedding)
        self.miss_content_mu, self.miss_content_log_var, self.miss_content_feat = self.content_model(self.missmatch_embedding.detach())

        interp_factor = torch.rand(1).to(self.device_map['img'])
        interp_factor = torch.clamp(interp_factor, 0.1, 0.95)
        self.interp_scontent = (1 - interp_factor) * self.style_feat + interp_factor * self.miss_style_feat

        images = self.generator(self.style_feat.to(self.device_map['img']).detach(), self.content_feat.to(self.device_map['img']))
        interp_o = self.generator(self.interp_scontent.to(self.device_map['img']).detach(), self.content_feat.to(self.device_map['img']))
        interp_c = self.generator(self.interp_scontent.to(self.device_map['img']).detach(), self.miss_content_feat.to(self.device_map['img']))

        self.fake_images = {}
        self.interp_images_o = {}
        self.interp_images_m = {}

        for idx, scale in enumerate(self.scales):
            self.fake_images[scale] = images[idx]
            self.interp_images_o[scale] = interp_o[idx]
            self.interp_images_m[scale] = interp_c[idx]

    def backward_discriminator(self):
        self.loss_real_c = 0.
        self.loss_real_u = 0.
        self.loss_fake_c = 0.
        self.loss_fake_u = 0.
        self.loss_real_m = 0.
        scale = self.scales[-1]  # interested only in final result, scales are kept for plotting reasons
        scores_real_u = self.discriminator(self.real_images[scale])
        scores_real_c = self.discriminator(self.real_images[scale], code=self.style_feat)
        scores_fake_u = self.discriminator(self.fake_images[scale].detach())
        scores_fake_c = self.discriminator(self.fake_images[scale].detach(), code=self.style_feat)
        scores_real_m = self.discriminator(self.real_images[scale], code=self.miss_style_feat)
        print(scores_real_u, scores_real_c, scores_fake_u)
        self.loss_real_u += self.generator_criterion(scores_real_u, target_is_real=True, for_discriminator=True) * self.cfg.lambda_uncond
        self.loss_real_c += self.generator_criterion(scores_real_c, target_is_real=True, for_discriminator=True) * self.cfg.lambda_match
        self.loss_real_m += self.generator_criterion(scores_real_m, target_is_real=False, for_discriminator=True) * self.cfg.lambda_match
        self.loss_fake_u += self.generator_criterion(scores_fake_u, target_is_real=False, for_discriminator=True) * self.cfg.lambda_uncond
        self.loss_fake_c += self.generator_criterion(scores_fake_c, target_is_real=False, for_discriminator=True) * self.cfg.lambda_match
        loss = self.loss_real_u + self.loss_real_c + self.loss_real_m + self.loss_fake_u + self.loss_fake_c
        loss *= 0.2

        self.loss_kld_style = self.distribution_criterion(self.style_mu, self.style_log_var) * self.cfg.lambda_kld + \
                              self.distribution_criterion(self.miss_style_mu, self.miss_style_log_var) * self.cfg.lambda_kld
        loss += self.loss_kld_style * 0.5
        loss.backward()
        #grad_penalty = self.cfg.lambda_gamma * 0.5 * discriminator_regularization(scores_real_u, self.real_images[scale],
        #                                                                          scores_fake_u, self.fake_images[scale])
        #grad_penalty.backward()

        # loss += discriminator_regularization(self.discriminator, self.real_images[scale], self.fake_images[scale], self.style_feat) * (
        #        self.cfg.lambda_gamma / 2)

    def backward_generator(self):
        loss = 0.
        self.loss_gen_u = 0.
        self.loss_gen_c = 0.
        self.loss_gen_i = 0.
        scale = self.scales[-1]
        pred_u = self.discriminator(self.fake_images[scale])
        pred_c = self.discriminator(self.fake_images[scale], code=self.style_feat.detach())
        pred_io = self.discriminator(self.interp_images_o[scale])
        pred_im = self.discriminator(self.interp_images_m[scale])
        pred_cio = self.discriminator(self.interp_images_o[scale], code=self.interp_scontent.detach())
        pred_cim = self.discriminator(self.interp_images_m[scale], code=self.interp_scontent.detach())
        self.loss_gen_u += self.generator_criterion(pred_u, target_is_real=True, for_discriminator=False) * self.cfg.lambda_uncond
        self.loss_gen_c += self.generator_criterion(pred_c, target_is_real=True, for_discriminator=False) * self.cfg.lambda_match
        self.loss_gen_i += self.generator_criterion(pred_io, target_is_real=True, for_discriminator=False) * self.cfg.lambda_interp
        self.loss_gen_i += self.generator_criterion(pred_im, target_is_real=True, for_discriminator=False) * self.cfg.lambda_interp
        self.loss_gen_i += self.generator_criterion(pred_cio, target_is_real=True, for_discriminator=False) * self.cfg.lambda_match
        self.loss_gen_i += self.generator_criterion(pred_cim, target_is_real=True, for_discriminator=False) * self.cfg.lambda_match
        loss += self.loss_gen_u + self.loss_gen_c + (self.loss_gen_i / 3)
        loss /= 3
        # self.loss_consistency = self.consistency_criterion(list(self.fake_images.values())) * self.cfg.lambda_clr + \
        #                        self.consistency_criterion(list(self.interp_images_o.values())) * self.cfg.lambda_clr + \
        #                        self.consistency_criterion(list(self.interp_images_m.values())) * self.cfg.lambda_clr
        self.loss_kld_content = self.distribution_criterion(self.content_mu, self.content_log_var) * self.cfg.lambda_kld + \
                                self.distribution_criterion(self.miss_content_mu, self.miss_content_log_var) * self.cfg.lambda_kld
        loss += self.loss_kld_content * 0.5
        loss.backward()

    def optimize(self):
        self.optim_generator.zero_grad()
        self.optim_content.zero_grad()
        self.set_requires_grad([self.generator], requires_grad=True)
        self.set_requires_grad([self.discriminator], requires_grad=False)
        self.forward_generator()
        self.backward_generator()
        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=10.)
        nn.utils.clip_grad_norm_(self.content_model.parameters(), max_norm=10.)

        self.optim_generator.step()
        self.optim_content.step()

        for _ in range(self.cfg.critic_iter):
            self.set_requires_grad([self.discriminator], requires_grad=True)
            self.set_requires_grad([self.generator], requires_grad=False)
            self.optim_style.zero_grad()
            self.optim_discriminator.zero_grad()
            #self.optim_language.zero_grad()
            self.backward_discriminator()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.)
            #nn.utils.clip_grad_norm_(self.language_model.parameters(), max_norm=10.)
            nn.utils.clip_grad_norm_(self.style_model.parameters(), max_norm=10.)
            self.optim_style.step()
            #self.optim_language.step()
            self.optim_discriminator.step()




        self.loss_loss = {'real_u': self.loss_real_u,
                          'real_c': self.loss_real_c,
                          'real_m': self.loss_real_m,
                          'fake_u': self.loss_fake_u,
                          'fake_c': self.loss_fake_c,
                          'gen_u': self.loss_gen_u,
                          'gen_c': self.loss_gen_c,
                          'gen_i': self.loss_gen_i,
                          'kld_style': self.loss_kld_style,
                          'kld_content': self.loss_kld_content,
                          }

        self.visual_dict['fake'] = self.fake_images

    # build dynamically sized layers, then resets all parameters
    def init_trainer_network(self):
        self.reset_parameters()
        self.gen_paramas = list(self.generator.parameters())
        self.dsc_params = list(self.discriminator.parameters())
        self.language_params = list(self.language_model.parameters())
        self.style_params = list(self.style_model.parameters())
        self.content_params = list(self.content_model.parameters())
        self.optim_language = AdamW(self.language_params, lr=self.cfg.lang_lr, betas=self.cfg.lang_betas, weight_decay=self.cfg.lang_wd)
        self.optim_generator = AdamW(self.gen_paramas, lr=self.cfg.gen_lr, betas=self.cfg.gen_betas, weight_decay=self.cfg.gen_wd)
        self.optim_discriminator = AdamW(self.dsc_params, lr=self.cfg.dsc_lr, betas=self.cfg.dsc_betas, weight_decay=self.cfg.dsc_wd)
        self.optim_style = AdamW(self.style_params, lr=self.cfg.lmf_lr, betas=self.cfg.lmf_betas, weight_decay=self.cfg.lmf_wd)
        self.optim_content = AdamW(self.content_params, lr=self.cfg.lmf_lr, betas=self.cfg.lmf_betas, weight_decay=self.cfg.lmf_wd)

    def get_current_visuals(self):
        dicts = super().get_current_visuals()
        image = self.visualizer.write_visuals(dicts)
        return image
