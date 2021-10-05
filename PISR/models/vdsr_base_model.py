from .vdsr_adder_net import *

# FIXME:

class VDSRBase(BaseNet):
    def __init__(self, scale, n_colors, d=12, s=12, m=4, k=1, vid_info=None,
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
    def __init__(self, scale, n_colors, d=56, s=56, m=4, k=1, vid_info=None,
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


def get_vdsr_adder_base_student(scale, n_colors, **kwargs):
    return VDSRAdder(scale, n_colors, **kwargs)
