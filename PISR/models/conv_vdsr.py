import torch
import torch.nn as nn
from collections import OrderedDict
from .base import BaseNet
from .encoder import get_encoder
import math


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


class ConvReLUBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, scale, n_colors, d=64, s=12, m=8, vdsr_weight_init=False):
        """

            input ---> input_layer ---> residual_layer ---> output_layer ---> output
          H*W*n_color   n_color->d        d->d                d->n_color       n_color

        :param scale:
        :param n_colors:
        :param d:
        :param s:
        :param m:
        :param vdsr_weight_init:
        """
        super(VDSR, self).__init__()
        self.scale = scale
        self.up_sampler = []
        self.up_sampler.append(nn.Sequential(*Upsampler(scale, n_colors, act=False)))
        self.input_layer = []
        self.input_layer.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
        self.output_layer = []
        self.output_layer.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=n_colors, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

        self.residual_layers = []
        for _ in range(m):
            self.residual_layers.append(nn.Sequential(
                ConvReLUBlock(in_channels=d, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

        net_list = []
        net_list.append(('upsampler', nn.Sequential(*self.up_sampler)))
        net_list.append(('input_layer', nn.Sequential(*self.input_layer)))

        for i in range(m // 2):
            net_list.append(('residual_layer_{}'.format(i), nn.Sequential(self.residual_layers[i*2], self.residual_layers[i*2+1])))

        net_list.append(('output_layer', nn.Sequential(*self.output_layer)))

        self.network = nn.Sequential(OrderedDict(net_list))

        if vdsr_weight_init:
            self.vdsr_weight_init()

    def vdsr_weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.network(x)


class VDSRAutoencoder(VDSR):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, k=1, encoder='inv_fsrcnn'):
        super(VDSRAutoencoder, self).__init__(scale, n_colors, d, s, m)

        self.encoder = get_encoder(encoder, scale=scale, d=d, s=s, k=k, n_colors=n_colors)

        net_list = []
        net_list.append(('encoder', nn.Sequential(*self.encoder)))
        net_list.append(('upsampler', nn.Sequential(*self.up_sampler)))
        net_list.append(('input_layer', nn.Sequential(*self.input_layer)))

        for i in range(m // 2):
            net_list.append(('residual_layer_{}'.format(i), nn.Sequential(self.residual_layers[i*2], self.residual_layers[i*2+1])))

        net_list.append(('output_layer', nn.Sequential(*self.output_layer)))

        self.network = nn.Sequential(OrderedDict(net_list))

    def forward(self, x):
        return self.network(x)


class VDSRTeacher(BaseNet):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, k=1, vid_info=None,
                 modules_to_freeze=None, initialize_from=None, modules_to_initialize=None,
                 encoder='lcscc'):
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

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name+'_mean'](x)
                var = self.vid_module_dict._modules[layer_name+'_var'](x)
                ret_dict[layer_name+'_mean'] = mean
                ret_dict[layer_name+'_var'] = var

        hr = x
        ret_dict['hr'] = hr
        return ret_dict


class VDSRStudent(BaseNet):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, vid_info=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, vdsr_weight_init=False):

        super(VDSRStudent, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = VDSR(scale, n_colors, d, s, m, vdsr_weight_init=vdsr_weight_init)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, LR, HR=None, teacher_pred_dict=None):
        ret_dict = dict()
        x = LR

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name+'_mean'](x)
                var = self.vid_module_dict._modules[layer_name+'_var'](x)
                ret_dict[layer_name+'_mean'] = mean
                ret_dict[layer_name+'_var'] = var

        hr = x
        ret_dict['hr'] = hr
        return ret_dict


# ------------------------------------ create self VDSR student model -------------------------
'''
    VDSRTeacher model:
    input[HR] ---> encoder[downSampling] ---> 
    VDSR model[ upsampler -> input_layer -> 
                residual_layer_0 [2 residual block] ->
                residual_layer_1 [2 residual block] ->
                residual_layer_2 [2 residual block] ->
                residual_layer_3 [2 residual block] ->
                output_layer] ---> output[SR]
                
    VDSRStudent model: we decrease the residual layer block
    input[HR] ---> encoder[downSampling] ---> 
    VDSR model[ upsampler -> input_layer -> 
                residual_layer_0 [1 residual block] ->
                residual_layer_1 [1 residual block] ->
                residual_layer_2 [1 residual block] ->
                residual_layer_3 [1 residual block] ->
                output_layer] ---> output[SR]
'''
class DecreaseVDSR(nn.Module):
    def __init__(self, scale, n_colors, d=32, s=12, m=4, vdsr_weight_init=False):
        """

            input ---> input_layer ---> residual_layer ---> output_layer ---> output
          H*W*n_color   n_color->d        d->d                d->n_color       n_color

        :param scale:
        :param n_colors:
        :param d:
        :param s:
        :param m:
        :param vdsr_weight_init:
        """
        super(DecreaseVDSR, self).__init__()
        self.scale = scale
        self.up_sampler = []
        self.up_sampler.append(nn.Sequential(*Upsampler(scale, n_colors, act=False)))
        self.input_layer = []
        self.input_layer.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
        self.output_layer = []
        self.output_layer.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=n_colors, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

        self.residual_layers = []
        for _ in range(m):
            self.residual_layers.append(nn.Sequential(
                ConvReLUBlock(in_channels=d, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

        net_list = []
        net_list.append(('upsampler', nn.Sequential(*self.up_sampler)))
        net_list.append(('input_layer', nn.Sequential(*self.input_layer)))

        # We decrease the residual layer num into half
        for i in range(m):
            net_list.append(('residual_layer_{}'.format(i), nn.Sequential(self.residual_layers[i])))

        net_list.append(('output_layer', nn.Sequential(*self.output_layer)))

        self.network = nn.Sequential(OrderedDict(net_list))

        if vdsr_weight_init:
            self.vdsr_weight_init()

    def vdsr_weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.network(x)


class DecreaseVDSRStudent(BaseNet):
    def __init__(self, scale, n_colors, d=32, s=12, m=4, vid_info=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, vdsr_weight_init=False):

        super(DecreaseVDSRStudent, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = DecreaseVDSR(scale, n_colors, d, s, m, vdsr_weight_init=vdsr_weight_init)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, LR, HR=None, teacher_pred_dict=None):
        ret_dict = dict()
        x = LR

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name+'_mean'](x)
                var = self.vid_module_dict._modules[layer_name+'_var'](x)
                ret_dict[layer_name+'_mean'] = mean
                ret_dict[layer_name+'_var'] = var

        hr = x
        ret_dict['hr'] = hr
        return ret_dict


# --------------------------------------- channel ---------------------------------------------------
class ChannelVDSR(nn.Module):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, vdsr_weight_init=False):
        """

            input ---> input_layer ---> residual_layer ---> output_layer ---> output
          H*W*n_color   n_color->d        d->d                d->n_color       n_color

        :param scale:
        :param n_colors:
        :param d:
        :param s:
        :param m:
        :param vdsr_weight_init:
        """
        super(ChannelVDSR, self).__init__()
        self.scale = scale
        self.up_sampler = []
        self.up_sampler.append(nn.Sequential(*Upsampler(scale, n_colors, act=False)))
        self.input_layer = []
        self.input_layer.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
        self.output_layer = []
        self.output_layer.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=n_colors, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

        self.residual_layers = []
        for _ in range(m):
            self.residual_layers.append(nn.Sequential(
                ConvReLUBlock(in_channels=d, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

        net_list = []
        net_list.append(('upsampler', nn.Sequential(*self.up_sampler)))
        net_list.append(('input_layer', nn.Sequential(*self.input_layer)))

        for i in range(m):
            net_list.append(('residual_layer_{}'.format(i), nn.Sequential(self.residual_layers[i])))

        net_list.append(('output_layer', nn.Sequential(*self.output_layer)))

        self.network = nn.Sequential(OrderedDict(net_list))

        if vdsr_weight_init:
            self.vdsr_weight_init()

    def vdsr_weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.network(x)





def get_vdsr_teacher(scale, n_colors, **kwargs):
    return VDSRTeacher(scale, n_colors, **kwargs)


def get_vdsr_student(scale, n_colors, **kwargs):
    return VDSRStudent(scale, n_colors, **kwargs)


def get_base_vdsr_student(scale, n_colors, **kwargs):
    return VDSRStudent(scale, n_colors, **kwargs)


# --------------------------------- decrease residual layer student model ----------
# this for student model initialization
def get_decrease_base_vdsr_student(scale, n_colors, **kwargs):
    return DecreaseVDSRStudent(scale, n_colors, **kwargs)


def get_decrease_vdsr_student(scale, n_colors, **kwargs):
    return DecreaseVDSRStudent(scale, n_colors, **kwargs)


def get_decrease_ll1_vdsr_student(scale, n_colors, **kwargs):
    return DecreaseVDSRStudent(scale, n_colors, **kwargs)


def get_decrease_ll2_vdsr_student(scale, n_colors, **kwargs):
    return DecreaseVDSRStudent(scale, n_colors, **kwargs)


def get_decrease_ll_vdsr_student(scale, n_colors, **kwargs):
    return DecreaseVDSRStudent(scale, n_colors, **kwargs)


