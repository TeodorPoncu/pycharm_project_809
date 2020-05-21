import torch
import os
import torch.nn as nn
from datetime import datetime
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.init as init


# some base template for models that has utility functions in it
class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device_ids = cfg.device_ids
        self.device = torch.device('cuda:{}'.format(self.device_ids[0])) if self.device_ids else torch.device('cpu')
        if self.device_ids:
            self.devices = [torch.device('cuda:{}'.format(i)) for i in self.device_ids]
        else:
            self.devices = [torch.device('cpu')]
        self.save_dir = os.path.join(cfg.checkpoint_dir,
                                     cfg.model_type + '_' +
                                     str(datetime.now())
                                     )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # networks that compose the model
        self.network_names = []
        self.device_map = {}

        # visualizer utils
        self.loss_names = []
        self.visual_names = []
        self.gradient_names = []

        # metric for learning rate scheduler
        self.metric = 0

    def update_learning_rate(self):
        pass

    def set_eval(self):
        for network_name in self.network_names:
            if isinstance(network_name, str):
                network = getattr(self, network_name)
                network.eval()

    def set_train(self):
        for network_name in self.network_names:
            if isinstance(network_name, str):
                network = getattr(self, network_name)
                network.train()

    def save_networks(self, iter):
        for name in self.network_names:
            if isinstance(name, str):
                save_filename = 'net_' + name + '_' + str(iter) +'.pth'
                save_path = os.path.join(self.save_dir, save_filename)
                network = getattr(self, name)
                torch.save(network.to('cpu').state_dict(), save_path)
                network.to(self.device_map[self.devices_name_map[name]])


    def init_weights(self, network, type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 0.0, gain)
                if hasattr(m, 'bias') and m.weight is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain)
                elif type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain)
                elif type == 'kaiming_normal':
                    init.kaiming_normal_(m.weight.data, 0.1, 'fan_in', 'leaky_relu')
                elif type == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight.data, 0.1, 'fan_in', 'leaky_relu')
                elif type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain)
                elif type == 'none':
                    m.reset_parameters()
            elif hasattr(m, 'weight_orig') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if type == 'normal':
                    init.normal_(m.weight_orig.data, 0.0, gain)
                elif type == 'xavier':
                    init.xavier_normal_(m.weight_orig.data, gain)
                elif type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight_orig.data, gain)
                elif type == 'kaiming_normal':
                    init.kaiming_normal_(m.weight_orig.data, 0.1, 'fan_in', 'leaky_relu')
                elif type == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight_orig.data, 0.1, 'fan_in', 'leaky_relu')
                elif type == 'orthogonal':
                    init.orthogonal_(m.weight_orig.data, gain)
                elif type == 'none':
                    m.reset_parameters()

        network.apply(init_func)

    def init_network(self, network, type='normal', gain=0.02, idx=0, device=None):
        if device is not None:
            network = network.to(device)
        else:
            network = network.to(self.device_ids[idx])
        #network = nn.DataParallel(network, self.device_ids)
        self.init_weights(network, type, gain)
        return network

    def set_requires_grad(self, networks, requires_grad=True):
        if not isinstance(networks, list):
            networks = [networks]
        for network in networks:
            if network is not None:
                for param in network.parameters():
                    param.requires_grad = requires_grad

    def get_current_visuals(self):
        # fetch visuals as a dictionary
        visuals = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visuals[name] = getattr(self, name)
        return visuals

    def get_current_losses(self):
        losses = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses[name] = getattr(self, 'loss_' + name)
        return losses

    def get_current_grads(self):

        def compute_grad_statistic(network):
            mean_grad = 0.0
            max_grad = 0.
            min_grad = 0.
            count = 0

            for param in network.parameters():
                if param.grad is not None:
                    grad_data = param.grad.data
                    mean_grad = mean_grad + torch.mean(torch.abs(grad_data))
                    min_grad = min(min_grad, torch.min(grad_data))
                    max_grad = max(max_grad, torch.max(grad_data))
                    count = count + 1
            if count > 0:
                mean_grad = mean_grad / count
            return {'mean': mean_grad, 'min': min_grad, 'max': max_grad}

        # this is going to be useful for Wasserstein
        grads = OrderedDict()
        for name in self.network_names:
            if isinstance(name, str):
                network = getattr(self, name)
                network_grads = compute_grad_statistic(network)
                grads[name] = network_grads
        return grads
