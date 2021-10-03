from .adder import adder2d
import torch.nn as nn
import torch
from collections import OrderedDict
from .encoder import get_encoder
from .base import BaseNet
import math


# -------------------------- AdderNet Layer -------------------------------- #
def Conv2d(in_channels, out_channels, kernel_size, stride, padding):
    """
        nn.Conv2d ---> adder.adder2d
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :return:
    """
    return adder2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return adder2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return adder2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Upsampler
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=False):
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


# -------------------------- AdderNet Layer -------------------------------- #
class ConvReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class AdderConvReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AdderConvReLUBlock, self).__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class VDSR(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=56, m=16, vdsr_weight_init=False):
        super(VDSR, self).__init__()
        self.scale = scale
        self.residual_layer = self.make_layer(ConvReLUBlock, m, d, s)
        self.input_layer = nn.Conv2d(in_channels=n_colors, out_channels=d, kernel_size=(3, 3,),
                                     stride=(1, 1), padding=(1, 1), bias=False)
        self.output_layer = nn.Conv2d(in_channels=d, out_channels=n_colors, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up_sampler = nn.Sequential(*Upsampler(scale, n_colors, act=False))

        self.network = nn.Sequential(OrderedDict(
            [
                ('input_layers', nn.Sequential(self.input_layer, self.relu)),
                ('residual_layers', nn.Sequential(self.residual_layer)),
                ('output_layers', nn.Sequential(self.output_layer))
            ]
        ))

        if vdsr_weight_init:
            self.fsrcnn_weight_init()

    def vdsr_weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()

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


class AdderVDSR(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=56, m=16, vdsr_weight_init=False):
        super(AdderVDSR, self).__init__()
        self.residual_layer = self.make_layer(AdderConvReLUBlock, m, d, s)
        self.input_layer = Conv2d(in_channels=n_colors, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.output_layer = Conv2d(in_channels=d, out_channels=n_colors, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.up_sampler = nn.Sequential(OrderedDict([
            ('upsampler', nn.Sequential(*Upsampler(scale, n_colors, act=False)))]))

        self.network = nn.Sequential(OrderedDict(
            [
                ('input_layers', nn.Sequential(self.input_layer, self.relu)),
                ('residual_layers', nn.Sequential(self.residual_layer)),
                ('output_layers', nn.Sequential(self.output_layer))
            ]
        ))

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


class VDSRAutoencoder(VDSR):
    def __init__(self, scale, n_colors, d=56, s=56, m=16, k=1, encoder='inv_fsrcnn'):
        super(VDSRAutoencoder, self).__init__(scale, n_colors, d, s, m)
        self.encoder = get_encoder(encoder, scale=scale, d=56, s=12, k=k, n_colors=n_colors)
        self.encoder_network = nn.Sequential(OrderedDict([('encoder', nn.Sequential(*self.encoder))]))

    def forward(self, x):
        x = self.encoder_network(x)
        x = self.up_sampler(x)
        residual = x
        out = self.network(x)
        out = torch.add(out, residual)
        return out


class VDSRTeacher(BaseNet):
    def __init__(self, scale, n_colors,  d=56, s=56, m=16, k=1, vid_info=None,
                 modules_to_freeze=None, initialize_from=None, modules_to_initialize=None,
                 encoder='inv_fsrcnn'):
        super(VDSRTeacher, self).__init__()

        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.modules_to_freeze = modules_to_freeze
        self.backbone = VDSRAutoencoder(scale, n_colors, d, s, m, k, encoder)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, HR, LR=None):
        ret_dict = dict()

        x = HR
        layer_names_encoder = self.backbone.encoder_network._modules.keys()

        for layer_name in layer_names_encoder:
            x = self.backbone.encoder_network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name+'_mean'](x)
                var = self.vid_module_dict._modules[layer_name+'_var'](x)
                ret_dict[layer_name+'_mean'] = mean
                ret_dict[layer_name+'_var'] = var


        layer_names_up_sampler = self.backbone.up_sampler._modules.keys()
        for layer_name in layer_names_up_sampler:
            x = self.backbone.up_sampler._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name + '_mean'](x)
                var = self.vid_module_dict._modules[layer_name + '_var'](x)
                ret_dict[layer_name + '_mean'] = mean
                ret_dict[layer_name + '_var'] = var

        residual = x

        layer_names_network = self.backbone.network._modules.keys()

        for layer_name in layer_names_network:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name + '_mean'](x)
                var = self.vid_module_dict._modules[layer_name + '_var'](x)
                ret_dict[layer_name + '_mean'] = mean
                ret_dict[layer_name + '_var'] = var

        x = torch.add(x, residual)
        hr = x
        ret_dict['hr'] = hr
        return ret_dict


class VDSRStudent(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=56, m=16, vid_info=None,
                modules_to_freeze=None, initialize_from=None,
                modules_to_initialize=None, vdsr_weight_init=False):

        super(VDSRStudent, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = AdderVDSR(scale, n_colors, d, s, m)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, HR, LR=None):
        ret_dict = dict()

        x = HR

        layer_names_up_sampler = self.backbone.up_sampler._modules.keys()
        for layer_name in layer_names_up_sampler:
            x = self.backbone.up_sampler._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name + '_mean'](x)
                var = self.vid_module_dict._modules[layer_name + '_var'](x)
                ret_dict[layer_name + '_mean'] = mean
                ret_dict[layer_name + '_var'] = var

        residual = x

        layer_names_network = self.backbone.network._modules.keys()

        for layer_name in layer_names_network:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name + '_mean'](x)
                var = self.vid_module_dict._modules[layer_name + '_var'](x)
                ret_dict[layer_name + '_mean'] = mean
                ret_dict[layer_name + '_var'] = var

        x = torch.add(x, residual)
        hr = x
        ret_dict['hr'] = hr
        return ret_dict


def get_vdsr_addernet_teacher(scale, n_colors, **kwargs):
    return VDSRTeacher(scale, n_colors, **kwargs)


def get_vdsr_addernet_student(scale, n_colors, **kwargs):
    return VDSRStudent(scale, n_colors, **kwargs)