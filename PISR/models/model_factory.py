import os, sys
import torch
from .fsrcnn import *
from .edsr_adder_net import *
from .fsrcnn_adder_net import *
# from .vdsr_adder_net import *
# from .vdsr_base_model import *
from .self_adder_fsrcnn import *
from .conv_vdsr import *
# from .self_adder_vdsr_base import *


device = None


def get_model(config, model_type):

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('model name:', config[model_type+'_model'].name)

    f = globals().get('get_' + config[model_type+'_model'].name)
    if config[model_type+'_model'].params is None:
        return f()
    else:
        return f(**config[model_type+'_model'].params)


def get_test_model(model_name, model_params):

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('model name:', model_name)

    f = globals().get('get_' + model_name)

    return f(**model_params)
