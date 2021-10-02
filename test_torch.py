from AdderNet.adder import adder2d
import torch.nn as nn
import torch
import math
from collections import OrderedDict


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
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))
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
        self.network_body = nn.Sequential(
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

        self.network = nn.Sequential()

    def forward(self, x):
        x = self.network_head(x)

        res = self.network_body(x)
        res = torch.add(x, res)

        x = self.network_tail(res)
        return self.network(x)