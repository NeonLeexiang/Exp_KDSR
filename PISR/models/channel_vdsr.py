from .conv_vdsr import *


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


class ChannelVDSRAutoencoder(ChannelVDSR):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, k=1, encoder='inv_fsrcnn'):
        super(ChannelVDSRAutoencoder, self).__init__(scale, n_colors, d, s, m)

        self.encoder = get_encoder(encoder, scale=scale, d=d, s=s, k=k, n_colors=n_colors)

        net_list = []
        net_list.append(('encoder', nn.Sequential(*self.encoder)))
        net_list.append(('upsampler', nn.Sequential(*self.up_sampler)))
        net_list.append(('input_layer', nn.Sequential(*self.input_layer)))

        for i in range(m):
            net_list.append(('residual_layer_{}'.format(i), nn.Sequential(self.residual_layers[i])))

        net_list.append(('output_layer', nn.Sequential(*self.output_layer)))

        self.network = nn.Sequential(OrderedDict(net_list))

    def forward(self, x):
        return self.network(x)


class ChannelVDSRTeacher(BaseNet):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, k=1, vid_info=None,
                 modules_to_freeze=None, initialize_from=None, modules_to_initialize=None,
                 encoder='lcscc'):
        super(ChannelVDSRTeacher, self).__init__()

        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.modules_to_freeze = modules_to_freeze
        self.backbone = ChannelVDSRAutoencoder(scale, n_colors, d, s, m, k, encoder)
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


class ChannelVDSRStudent(BaseNet):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, vid_info=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, vdsr_weight_init=False):

        super(ChannelVDSRStudent, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = ChannelVDSR(scale, n_colors, d, s, m, vdsr_weight_init=vdsr_weight_init)
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


# --------------------------------------- Encoder Decoder ---------------------------------------------------
class EnDeChannelVDSR(nn.Module):
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
        super(EnDeChannelVDSR, self).__init__()
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
                nn.Conv2d(in_channels=d, out_channels=d//2, kernel_size=(1, 1), stride=(1, 1)),
                ConvReLUBlock(in_channels=d//2, out_channels=d//2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(in_channels=d//2, out_channels=d, kernel_size=(1, 1), stride=(1, 1))
            ))

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


class EnDeChannelVDSRStudent(BaseNet):
    def __init__(self, scale, n_colors, d=32, s=12, m=8, vid_info=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, vdsr_weight_init=False):

        super(EnDeChannelVDSRStudent, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = EnDeChannelVDSR(scale, n_colors, d, s, m, vdsr_weight_init=vdsr_weight_init)
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


# --------------------------------------------
def get_channel_vdsr_student(scale, n_colors, **kwargs):
    return ChannelVDSRStudent(scale, n_colors, **kwargs)


def get_channel_vdsr_teacher(scale, n_colors, **kwargs):
    return ChannelVDSRTeacher(scale, n_colors, **kwargs)


def get_base_channel_vdsr_student(scale, n_colors, **kwargs):
    return ChannelVDSRStudent(scale, n_colors, **kwargs)


def get_base_de_channel_vdsr_student(scale, n_colors, **kwargs):
    return ChannelVDSRStudent(scale, n_colors, d=16, **kwargs)


def get_de_channel_vdsr_student(scale, n_colors, **kwargs):
    return ChannelVDSRStudent(scale, n_colors, d=16, **kwargs)


def get_ll1_de_channel_vdsr_student(scale, n_colors, **kwargs):
    return ChannelVDSRStudent(scale, n_colors, d=16, **kwargs)


def get_ll2_de_channel_vdsr_student(scale, n_colors, **kwargs):
    return ChannelVDSRStudent(scale, n_colors, d=16, **kwargs)


def get_ll_de_channel_vdsr_student(scale, n_colors, **kwargs):
    return ChannelVDSRStudent(scale, n_colors, d=16, **kwargs)


# -----------
def get_base_de_ende_channel_vdsr_student(scale, n_colors, **kwargs):
    return EnDeChannelVDSRStudent(scale, n_colors, d=32, **kwargs)


def get_de_ende_channel_vdsr_student(scale, n_colors, **kwargs):
    return EnDeChannelVDSRStudent(scale, n_colors, d=32, **kwargs)


def get_ll1_de_ende_channel_vdsr_student(scale, n_colors, **kwargs):
    return EnDeChannelVDSRStudent(scale, n_colors, d=32, **kwargs)


def get_ll2_de_ende_channel_vdsr_student(scale, n_colors, **kwargs):
    return EnDeChannelVDSRStudent(scale, n_colors, d=32, **kwargs)


def get_ll_de_ende_channel_vdsr_student(scale, n_colors, **kwargs):
    return EnDeChannelVDSRStudent(scale, n_colors, d=32, **kwargs)


# def get_channel_ll1_vdsr_student(scale, n_colors, **kwargs):
#     return DecreaseVDSRStudent(scale, n_colors, **kwargs)
#
#
# def get_channel_ll2_vdsr_student(scale, n_colors, **kwargs):
#     return DecreaseVDSRStudent(scale, n_colors, **kwargs)
#
#
# def get_channel_ll_vdsr_student(scale, n_colors, **kwargs):
#     return DecreaseVDSRStudent(scale, n_colors, **kwargs)
