import torch
import torch.nn as nn
from .adder4cuda import Adder2D as adder2d_cuda
from .adder import adder2d
from .self_act import PowerActivation


# -------------------------- AdderNet Layer -------------------------------- #
def conv2d(in_channels, out_channels, kernel_size, stride, padding):
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


class AdderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AdderLayer, self).__init__()
        self.conv = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding)
        self.act = PowerActivation()
        self.batch_normal = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        res = x
        x = self.conv(x)
        x += res
        x = self.batch_normal(x)
        x = self.act(x)
        return x


class SelfAdderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SelfAdderLayer, self).__init__()

        self.net = []
        for c in range(3):
            self.net.append(nn.Sequential(AdderLayer(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding)))
        self.network = nn.Sequential(*self.net)
        # self.act = PowerActivation()
        # self.batch_normal = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # res = x
        # x = self.conv(x)
        # x += res
        # x = self.batch_normal(x)
        # x = self.act(x)
        return self.network(x)


# -------------------------- CUDA AdderNet Layer -------------------------------- #
def conv2d_cuda(in_channels, out_channels, kernel_size, stride, padding):
    """
        nn.Conv2d ---> adder.adder2d
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :return:
    """
    return adder2d_cuda(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class AdderLayerCUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AdderLayerCUDA, self).__init__()
        self.conv = conv2d_cuda(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.act = PowerActivation()
        self.batch_normal = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        res = x
        x = self.conv(x)
        x += res
        x = self.batch_normal(x)
        return self.act(x)


class SelfAdderLayerCUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SelfAdderLayerCUDA, self).__init__()

        self.net = []
        for c in range(3):
            self.net.append(nn.Sequential(AdderLayerCUDA(in_channels=in_channels, out_channels=out_channels,
                                                         kernel_size=kernel_size, stride=stride, padding=padding)))
        self.network = nn.Sequential(*self.net)
        # self.act = PowerActivation()
        # self.batch_normal = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # res = x
        # x = self.conv(x)
        # x += res
        # x = self.batch_normal(x)
        # x = self.act(x)
        return self.network(x)
