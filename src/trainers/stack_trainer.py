from ..networks.generators_v1 import *
from ..networks.language_models import *
from ..networks.vae import *
from ..networks.base_model import *
from ..networks.discriminator_v1 import HingeJointConvolutionalDiscriminator, HingeLoss
from torch.optim.adamw import AdamW
import torchvision.utils as vutils


def KL_Loss(mu, log_var):
    KLD = mu.pow_(2).add_(log_var.exp_()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.mean(KLD).mul_(-0.5)
    return KLD


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
        self.features_model = VAE(self.language_model.out_size(), self.device_map['feat'], cfg).to(self.device_map['feat'])

        gen_0 = VanillaGenerator(self.device_map['img'], cfg)
        gen_1 = ResidualGenerator(self.device_map['img'], cfg, 8)
        gen_2 = ResidualGenerator(self.device_map['img'], cfg, 4)

        img_0 = ImageGenerator(self.device_map['img'], 8 * cfg.ngf, cfg)
        img_1 = ImageGenerator(self.device_map['img'], 4 * cfg.ngf, cfg)
        img_2 = ImageGenerator(self.device_map['img'], 2 * cfg.ngf, cfg)

        self.visualizer = Visualizer(cfg)
        self.criterion = HingeLoss().to(self.device_map['img'])

        gens = [gen_0, gen_1, gen_2]
        imgs = [img_0, img_1, img_2]

        self.visual_names = ['visual_dict']
        self.loss_names = ['dict']

        self.visual_dict = {'real': None, 'fake': None}

        self.generators = {}
        self.discriminators = {}
        for scale, gen, img in zip(self.scales, gens, imgs):
            self.generators[scale] = nn.ModuleDict({'features': gen, 'image': img})
            self.discriminators[scale] = HingeJointConvolutionalDiscriminator(self.device_map['img'], cfg)

        self.generators = nn.ModuleDict(self.generators).to(self.device_map['img'])
        self.discriminators = nn.ModuleDict(self.discriminators).to(self.device_map['img'])
        self.consistency_criterion = ColorConsistencyLoss(cfg).to(self.device_map['img'])

    def reset_parameters(self):
        for scale in self.generators:
            self.init_weights(self.generators[scale]['features'], self.cfg.gen_init, self.cfg.gen_gain)
            self.init_weights(self.generators[scale]['image'], self.cfg.gen_init, self.cfg.gen_gain)
        self.init_weights(self.features_model, self.cfg.gen_init, self.cfg.gen_gain)
        if self.language_model.trainable:
            if self.cfg.embbedings_pretrained:
                self.init_weights(self.language_model, self.cfg.gen_init, self.gen_gain)

    def set_input(self, input):
        self.input = {}
        self.input['text'] = input['text']
        self.input['images'] = {}
        self.visual_dict['real'] = input['images']
        for scale in input['images']:
            self.input['images'][scale] = input['images'][scale].to(self.devices[0])

    def forward_discriminator(self, input, condition, detach=False):
        pred = {}
        for scale in self.scales:
            if detach:
                pred[scale] = self.discriminators[scale](input[scale].detach(), condition)
            else:
                pred[scale] = self.discriminators[scale](input[scale], condition)
        return pred

    def backward_discriminator(self, pred_real, pred_fake):
        loss = {}
        loss_dsc = 0.

        for scale in self.scales:
            loss[scale] = self.discriminators[scale].compute_loss(pred_real[scale], pred_fake[scale])
            loss_dsc += (loss[scale]['u_fake'] + loss[scale]['u_real']) * self.cfg.lambda_dsc_u + \
                        (loss[scale]['c_fake'] + loss[scale]['c_real']) * self.cfg.lambda_dsc_c
        loss_dsc *= 0.5
        loss_dsc.backward()
        return loss_dsc

    def forward_generator(self):
        embeddings = self.language_model(self.input['text'])
        self.mu, self.log_var, embeddings = self.features_model(embeddings.to(self.device_map['feat']))
        embeddings = embeddings.to(self.device_map['img'])

        self.condition = embeddings

        results = {}
        features = {}
        for idx, scale in enumerate(self.scales):
            if idx == 0:
                features[scale] = self.generators[scale]['features'](embeddings)
                results[scale] = self.generators[scale]['image'](features[scale])
            else:
                features[scale] = self.generators[scale]['features'](features[self.scales[idx - 1]])
                results[scale] = self.generators[scale]['image'](features[scale])

        return results

    def backward_generator(self, images):
        loss_gen = 0.
        pred = self.forward_discriminator(images, self.condition, detach=False)
        for scale in pred:
            loss_gen += -torch.mean(pred[scale]['u']) * self.cfg.lambda_gen_u + \
                        -torch.mean(pred[scale]['c']) * self.cfg.lambda_gen_c
        loss_consistency = self.consistency_criterion(list(images.values())) * self.cfg.lambda_clr
        loss_kld = KL_Loss(self.mu, self.log_var) * self.cfg.lambda_kld
        loss = loss_gen + loss_consistency + loss_kld
        loss.backward()
        return loss_gen, loss_consistency, loss_kld

    def optimize(self):
        self.optim_generator.zero_grad()
        if self.language_model.trainable:
            self.optim_language.zero_grad()
        self.set_requires_grad([self.discriminators], requires_grad=False)
        self.visuals = self.forward_generator()
        self.loss_gen, self.loss_consistency, self.loss_kld = self.backward_generator(self.visuals)
        print('kld:', self.loss_kld)
        self.optim_generator.step()
        if self.language_model.trainable:
            self.optim_language.step()

        self.set_requires_grad([self.discriminators], requires_grad=True)
        for _ in range(self.cfg.critic_iter):
            #print('stepping in dsc')
            self.optim_discriminator.zero_grad()
            pred_real = self.forward_discriminator(self.input['images'], self.condition.detach(), detach=False)
            pred_fake = self.forward_discriminator(self.visuals, self.condition.detach(), detach=True)
            self.loss_dsc = self.backward_discriminator(pred_real, pred_fake)
            self.optim_discriminator.step()

        self.loss_dict = {'dsc': self.loss_dsc, 'gen':self.loss_gen, 'consistency':self.loss_consistency, 'kld':self.loss_kld}
        #print(self.loss_dict)
        self.visual_dict['fake'] = self.visuals

    # build dynamically sized layers, then resets all parameters
    def init_trainer_network(self):
        images = self.forward_generator()
        _ = self.forward_discriminator(images, self.condition)
        self.reset_parameters()

        gen_paramas = list(self.generators.parameters()) + list(self.features_model.parameters())
        dsc_params = list(self.discriminators.parameters())
        self.optim_generator = AdamW(gen_paramas, lr=self.cfg.gen_lr, betas=self.cfg.gen_betas, weight_decay=self.cfg.gen_wd)

        self.optim_discriminator = AdamW(dsc_params, lr=self.cfg.dsc_lr, betas=self.cfg.dsc_betas,
                                         weight_decay=self.cfg.dsc_wd)

        # TODO add to yaml params for language optimizer
        if self.language_model.trainable:
            self.optim_language = AdamW(list(self.language_model.parameters()), lr=self.cfg.gen_lr, betas=self.cfg.gen_betas,
                                    weight_decay=self.cfg.gen_wd)

    def get_current_visuals(self):
        dicts = super().get_current_visuals()
        image = self.visualizer.write_visuals(dicts)
        return image