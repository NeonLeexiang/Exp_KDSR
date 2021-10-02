from .adder import adder2d
import torch.nn as nn
import torch
import math
from collections import OrderedDict
from .encoder import get_encoder
from .base import BaseNet


# -------------------------- AdderNet Layer -------------------------------- #
def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
    """
        nn.Conv2d ---> adder.adder2d
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :return:
    """
    return adder2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
# -------------------------- AdderNet Layer -------------------------------- #


# -------------------------- EDSR ------------------------------------------ #
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1), padding=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, padding, bias=False, bn=False, act=nn.ReLU(inplace=True), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = []

        for i in range(2):
            m.append(conv2d(in_channels=n_feats, out_channels=n_feats,
                            kernel_size=kernel_size, stride=1, padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=False):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv2d(in_channels=n_feats, out_channels=4 * n_feats,
                                   kernel_size=3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv2d(in_channels=n_feats, out_channels=9 * n_feats,
                               kernel_size=3, padding=1, bias=bias))
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


class EDSR(nn.Module):
    def __init__(self, scale, n_colors, d=256, s=256, m=32, res_scale=0.1, rgb_range=1):
        super(EDSR, self).__init__()

        self.n_channels = n_colors
        self.n_resblocks = m
        self.n_feats = d
        self.scale = scale
        self.res_scale = res_scale
        self.rgb_range = rgb_range

        self.kernel_size = 3
        self.padding = 1
        self.act = nn.ReLU(True)

        self.sub_mean = []
        self.sub_mean.append(nn.Sequential(MeanShift(self.rgb_range)))

        self.add_mean = []
        self.add_mean.append(nn.Sequential(MeanShift(self.rgb_range, sign=1)))

        self.net_head = []
        self.net_head.append(nn.Sequential(conv2d(self.n_channels, self.n_feats, kernel_size=self.kernel_size, padding=self.padding)))

        self.net_body = []
        net_body = [
            ResBlock(
                n_feats=self.n_feats, kernel_size=self.kernel_size, padding=self.padding,
                act=self.act, res_scale=self.res_scale
            ) for _ in range(self.n_resblocks)
        ]
        net_body.append(conv2d(in_channels=self.n_feats, out_channels=self.n_feats, kernel_size=self.kernel_size, padding=self.padding))
        self.net_body.append(nn.Sequential(*net_body))

        self.net_tail = []
        net_tail = [
            Upsampler(self.scale, self.n_feats, act=False),
            nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_channels,
                      kernel_size=self.kernel_size, padding=self.padding)
        ]
        self.net_tail.append(nn.Sequential(*net_tail))

        self.network_head = nn.Sequential(
            OrderedDict([
                ('sub_mean', nn.Sequential(*self.sub_mean)),
                ('net_head', nn.Sequential(*self.net_head)),
            ])
        )
        self.network = nn.Sequential(
            OrderedDict([
                ('net_body', nn.Sequential(*self.net_body)),
            ])
        )
        self.network_tail = nn.Sequential(
            OrderedDict([
                ('net_tail', nn.Sequential(*self.net_tail)),
                ('add_mean', nn.Sequential(*self.add_mean)),
            ])
        )

    def forward(self, x):
        x = self.network_head(x)

        res = self.network(x)
        res = torch.add(x, res)

        x = self.network_tail(res)
        return x


class EDSRAutoencoder(EDSR):
    def __init__(self, scale, n_colors, d=256, s=256, m=32, res_scale=0.1, rgb_range=1, k=1, encoder='inv_fsrcnn'):
        super(EDSRAutoencoder, self).__init__(scale, n_colors, d, s, m, res_scale, rgb_range)

        self.encoder = get_encoder(encoder, scale=scale, d=56, s=12, k=k, n_colors=n_colors)

        self.network_encoder = nn.Sequential(
            OrderedDict([
                ('encoder', nn.Sequential(*self.encoder)),
            ])
        )

    def forward(self, x):
        x = self.network_encoder(x)
        x = self.network_head(x)

        res = self.network(x)
        res = torch.add(x, res)

        x = self.network_tail(res)
        return x


class EDSRStudent(BaseNet):
    def __init__(self, scale, n_colors, d=32, s=32, m=32, res_scale=0.1, rgb_range=1, vid_info=None, modules_to_freeze=None, initialize_from=None, modules_to_initialize=None):
        super(EDSRStudent, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = EDSR(scale, n_colors, d, s, m, res_scale, rgb_range)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, LR, HR=None, teacher_pred_dict=None):
        ret_dict = dict()
        x = LR

        layer_names_head = self.backbone.network_head._modules.keys()
        for layer_name in layer_names_head:
            x = self.backbone.network_head._modules[layer_name](x)
            ret_dict[layer_name] = x

        res = x

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            res = self.backbone.network._modules[layer_name](res)
            ret_dict[layer_name] = res
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name + '_mean'](res)
                var = self.vid_module_dict._modules[layer_name + '_var'](res)
                ret_dict[layer_name + '_mean'] = mean
                ret_dict[layer_name + '_var'] = var

        res = torch.add(x, res)

        layer_names_tail = self.backbone.network_tail._modules.keys()
        for layer_name in layer_names_tail:
            x = self.backbone.network_tail._modules[layer_name](res)
            ret_dict[layer_name] = x

        hr = x
        ret_dict['hr'] = hr
        return ret_dict


class EDSRTeacher(BaseNet):
    def __init__(self, scale, n_colors, d=16, s=16, m=32, res_scale=0.1, rgb_range=1, k=1, vid_info=None, modules_to_freeze=None, initialize_from=None, modules_to_initialize=None, encoder='inv_fsrcnn'):
        super(EDSRTeacher, self).__init__()

        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.modules_to_freeze = modules_to_freeze
        self.backbone = EDSRAutoencoder(scale, n_colors, d, s, m, res_scale, rgb_range, encoder=encoder)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, HR, LR=None):
        ret_dict = dict()

        x = HR

        layer_names_encoder = self.backbone.network_encoder._modules.keys()
        for layer_name in layer_names_encoder:
            x = self.backbone.network_encoder._modules[layer_name](x)
            ret_dict[layer_name] = x

        layer_names_head = self.backbone.network_head._modules.keys()
        for layer_name in layer_names_head:
            x = self.backbone.network_head._modules[layer_name](x)
            ret_dict[layer_name] = x

        res = x

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            res = self.backbone.network._modules[layer_name](res)
            ret_dict[layer_name] = res
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name+'_mean'](res)
                var = self.vid_module_dict._modules[layer_name+'_var'](res)
                ret_dict[layer_name+'_mean'] = mean
                ret_dict[layer_name+'_var'] = var

        res = torch.add(x, res)

        layer_names_tail = self.backbone.network_tail._modules.keys()
        for layer_name in layer_names_tail:
            x = self.backbone.network_tail._modules[layer_name](res)
            ret_dict[layer_name] = x

        hr = x
        ret_dict['hr'] = hr
        return ret_dict


def get_edsr_teacher(scale, n_colors, **kwargs):
    return EDSRTeacher(scale, n_colors, **kwargs)


def get_edsr_student(scale, n_colors, **kwargs):
    return EDSRStudent(scale, n_colors, **kwargs)


