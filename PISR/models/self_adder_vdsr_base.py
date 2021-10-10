import torch
import torch.nn as nn
from .self_adder import AdderLayerCUDA
from collections import OrderedDict
import math
from .base import BaseNet


# Upsampler
class Upsampler(nn.Sequential):
    # FIXME: setting bias=True
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats,
                                   kernel_size=(3, 3), padding=(1, 1), bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(in_channels=n_feats, out_channels=9 * n_feats,
                               kernel_size=(3, 3), padding=(1, 1), bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class SelfAdderConvReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SelfAdderConvReLUBlock, self).__init__()
        self.conv = AdderLayerCUDA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class AdderVDSR(nn.Module):
    def __init__(self, scale, n_colors, d=64, s=64, m=16, vdsr_weight_init=False):
        super(AdderVDSR, self).__init__()
        self.residual_layer = []
        self.residual_layer.append(self.make_layer(SelfAdderConvReLUBlock, m, d, d))
        self.input_layer = []
        self.input_layer.append(nn.Conv2d(in_channels=n_colors, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.output_layer = []
        self.output_layer.append(nn.Conv2d(in_channels=d, out_channels=n_colors, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.input_layer.append(nn.ReLU(inplace=True))

        self.up_sampler = nn.Sequential(OrderedDict([
            ('upsampler', nn.Sequential(*Upsampler(scale, n_colors, act=False)))]))

        self.network = nn.Sequential(OrderedDict(
            [
                ('input_layers', nn.Sequential(*self.input_layer)),
                ('residual_layers', nn.Sequential(*self.residual_layer)),
                ('output_layers', nn.Sequential(*self.output_layer))
            ]
        ))
        # FIXME: init
        # self.vdsr_adder_weight_init()

    def forward(self, x):
        x = self.up_sampler(x)
        residual = x
        out = self.network(x)
        out = torch.add(out, residual)
        return out

    def make_layer(self, block, num_of_layer, in_channels, out_channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def vdsr_adder_weight_init(self):
        for name, param in self.network.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            if 'bias' in name:
                nn.init.normal_(param, 0, 0.001)


class VDSRBase(BaseNet):
    def __init__(self, scale, n_colors, d=64, s=64, m=16, k=1, vid_info=None,
                modules_to_freeze=None, initialize_from=None,
                modules_to_initialize=None, encoder=None, vdsr_weight_init=False):

        super(VDSRBase, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = VDSR(scale, n_colors, d, s, m)
        self.vid_info = None
        self.vid_module_dict = None

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, LR, HR=None):
        ret_dict = dict()

        x = LR

        layer_names_up_sampler = self.backbone.up_sampler._modules.keys()
        for layer_name in layer_names_up_sampler:
            x = self.backbone.up_sampler._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual = x

        layer_names_network = self.backbone.network._modules.keys()

        for layer_name in layer_names_network:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        x = torch.add(x, residual)
        hr = x
        ret_dict['hr'] = hr
        return ret_dict


class VDSRAdder(BaseNet):
    def __init__(self, scale, n_colors, d=64, s=64, m=16, k=1, vid_info=None,
                modules_to_freeze=None, initialize_from=None,
                modules_to_initialize=None, encoder=None, vdsr_weight_init=False):

        super(VDSRAdder, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = AdderVDSR(scale, n_colors, d, s, m)
        self.vid_info = None
        self.vid_module_dict = None

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, LR, HR=None):
        ret_dict = dict()

        x = LR

        layer_names_up_sampler = self.backbone.up_sampler._modules.keys()
        for layer_name in layer_names_up_sampler:
            x = self.backbone.up_sampler._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual = x

        layer_names_network = self.backbone.network._modules.keys()

        for layer_name in layer_names_network:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        x = torch.add(x, residual)
        hr = x
        ret_dict['hr'] = hr
        return ret_dict


def get_vdsr_base_student(scale, n_colors, **kwargs):
    return VDSRBase(scale, n_colors, **kwargs)


def get_self_adder_vdsr_student(scale, n_colors, **kwargs):
    return VDSRAdder(scale, n_colors, **kwargs)
