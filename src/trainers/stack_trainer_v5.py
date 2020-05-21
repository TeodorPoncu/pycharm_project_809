from ..networks.generators_v5 import *
from ..networks.language_models import *
from ..networks.vae import *
from ..networks.base_model import *
from ..networks.discriminator_v4 import HingeJointPatchConvolutionalDiscriminator, HingeLoss
from torch.optim.adamw import AdamW
import torchvision.utils as vutils
from random import randint, uniform


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


class StackTrainer(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scales = [str((cfg.load_size // (2 ** i))) for i in range(3)]
        self.scales.reverse()

        self.device_map = {'text': self.devices[0], 'feat': self.devices[0], 'img': self.devices[0]}
        self.network_names = ['language_model', 'features_model', 'generators', 'discriminators']
        self.device_name_map = {'language_model': 'text',
                                'features_model': 'feat',
                                'generators': 'img',
                                'discriminators': 'img'}

        self.language_model = GPT2(self.device_map['text'], cfg).to(self.device_map['text'])
        self.features_model = GLUVAE(self.language_model.out_size(), self.device_map['feat'], cfg).to(self.device_map['feat'])

        self.visualizer = Visualizer(cfg)
        self.criterion = HingeLoss().to(self.device_map['img'])

        self.visual_names = ['visual_dict']
        self.loss_names = ['loss']

        self.visual_dict = {'real': None, 'fake': None}
        self.image_pool = {scale: ImagePool(cfg) for scale in self.scales}

        self.generator = StyleGenerator(self.device_map['img'], cfg).to(self.device_map['img'])
        self.discriminators = {}
        for scale in self.scales:
            self.discriminators[scale] = HingeJointPatchConvolutionalDiscriminator(int(scale), self.device_map['img'], cfg)

        self.discriminators = nn.ModuleDict(self.discriminators).to(self.device_map['img'])
        self.consistency_criterion = ColorConsistencyLoss(cfg).to(self.device_map['img'])
        self.distribution_criterion = KLDLoss().to(self.device_map['img'])
        self.reconstruction_criterion = nn.L1Loss().to(self.device_map['img'])

    def reset_parameters(self):
        self.init_weights(self.generator, self.cfg.gen_init, self.cfg.gen_gain)
        self.init_weights(self.features_model, self.cfg.gen_init, self.cfg.gen_gain)
        self.init_weights(self.discriminators, self.cfg.dsc_init, self.cfg.dsc_gain)
        if self.language_model.trainable:
            if not self.cfg.embeddings_pretrained:
                self.init_weights(self.language_model, self.cfg.gen_init, self.gen_gain)

    def set_input(self, input):
        self.real_images = {}
        self.text_input = input['text']
        self.visual_dict['real'] = input['images']
        for scale in input['images']:
            self.real_images[scale] = input['images'][scale].to(self.devices[0])

    def forward_discriminator(self, input, code=None):
        pred = {}
        for scale in self.scales:
            pred[scale] = self.discriminators[scale](input[scale], code)
        return pred

    def backward_discriminator(self, conditioned_images, condition):
        loss = 0.
        self.loss_dsc_u = {}
        self.loss_dsc_c = {}
        for scale in self.scales:
            real_images = self.image_pool[scale].fetch_real()
            fake_images = self.image_pool[scale].fetch_fake()
            images = torch.cat([real_images, fake_images], dim=0)
            scores_u = self.discriminators[scale](images)
            scores_c = self.discriminators[scale](conditioned_images[scale], condition)
            self.loss_dsc_u[scale] = self.discriminators[scale].compute_loss(scores_u, conditioned=False)
            self.loss_dsc_c[scale] = self.discriminators[scale].compute_loss(scores_c, conditioned=True)
            loss += self.loss_dsc_u[scale] * self.cfg.lambda_dsc_u + self.loss_dsc_c[scale] * self.cfg.lambda_dsc_c
        loss *= 0.5
        loss.backward()

    def forward_generator(self):
        embeddings = self.language_model(self.text_input)
        self.mu, self.log_var, embeddings = self.features_model(embeddings.to(self.device_map['feat']))
        embeddings = embeddings.to(self.device_map['img'])
        self.condition = embeddings
        self.z = torch.cat([self.condition, torch.randn_like(self.condition)], dim=1)

        result = self.generator(self.condition, self.z)
        results = {}

        for idx, scale in enumerate(self.scales):
            results[scale] = result[idx]
            self.image_pool[scale].update(real_images=self.real_images[scale], fake_images=results[scale].detach())

        return results

    def backward_generator(self, images):
        loss = 0.
        u_pred = self.forward_discriminator(images)
        c_pred = self.forward_discriminator(images, code=self.condition)
        self.loss_gen = {}
        for scale in self.scales:
            self.loss_gen[scale] = -torch.mean(u_pred[scale]) * self.cfg.lambda_gen_u + \
                                   -torch.mean(c_pred[scale]) * self.cfg.lambda_gen_c
            loss += self.loss_gen[scale]
        self.loss_consistency = self.consistency_criterion(list(images.values())) * self.cfg.lambda_clr
        self.loss_kld = self.distribution_criterion(self.mu, self.log_var) * self.cfg.lambda_kld
        loss += self.loss_consistency + self.loss_kld
        loss.backward(retain_graph=True)

    def optimize(self):
        self.optim_generator.zero_grad()
        if self.language_model.trainable:
            self.optim_feature.zero_grad()
        self.set_requires_grad([self.discriminators], requires_grad=False)
        self.fake_images = self.forward_generator()
        self.backward_generator(self.fake_images)
        self.optim_generator.step()

        self.set_requires_grad([self.discriminators], requires_grad=True)
        conditioned_images = {scale: torch.cat([self.real_images[scale], self.fake_images[scale].detach()], dim=0) for scale in self.scales}
        for _ in range(self.cfg.critic_iter):
            self.optim_discriminator.zero_grad()
            self.backward_discriminator(conditioned_images, self.condition)
            self.optim_discriminator.step()
        self.optim_feature.step()

        self.loss_loss = {'dsc_c': self.loss_dsc_c, 'dsc_u': self.loss_dsc_u, 'gen': self.loss_gen,
                          'consistency': self.loss_consistency, 'kld': self.loss_kld}
        self.visual_dict['fake'] = self.fake_images
        # print(self.loss_dict)


    # build dynamically sized layers, then resets all parameters
    def init_trainer_network(self):
        images = self.forward_generator()
        _ = self.forward_discriminator(images, self.condition)
        self.reset_parameters()

        gen_paramas = list(self.generator.parameters())
        dsc_params = list(self.discriminators.parameters())
        self.optim_generator = AdamW(gen_paramas, lr=self.cfg.gen_lr, betas=self.cfg.gen_betas, weight_decay=self.cfg.gen_wd)

        self.optim_discriminator = AdamW(dsc_params, lr=self.cfg.dsc_lr, betas=self.cfg.dsc_betas,
                                         weight_decay=self.cfg.dsc_wd)

        # TODO add to yaml params for language optimizer
        feature_params = list(self.features_model.parameters())
        if self.language_model.trainable:
            feature_params = feature_params + list(self.language_model.parameters())
        self.optim_feature = AdamW(feature_params, lr=self.cfg.gen_lr, betas=self.cfg.gen_betas, weight_decay=self.cfg.gen_wd)

    def get_current_visuals(self):
        dicts = super().get_current_visuals()
        image = self.visualizer.write_visuals(dicts)
        return image
