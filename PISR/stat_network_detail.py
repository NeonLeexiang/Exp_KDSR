import torch
import torch.nn as nn
import torch.nn.functional as F

# from ptflops import get_model_complexity_info
# from models.self_adder_fsrcnn import get_self_fsrcnn_student, get_self_base_fsrcnn_student
#
#
# if __name__ == '__main__':
#     conv_model = get_self_base_fsrcnn_student(2, 1)
#     adder_model = get_self_fsrcnn_student(2, 1)
#
#     macs, params = get_model_complexity_info(conv_model, (1, 32, 32), as_strings=True,
#                                              print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#
#     print('=' * 50)
#     print('#' * 50)
#     print('=' * 50)
#
#     macs, params = get_model_complexity_info(adder_model, (1, 32, 32), as_strings=True,
#                                              print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#
#





# from thop import profile
# from thop import clever_format
# from models.self_adder_fsrcnn import get_self_fsrcnn_student, get_self_base_fsrcnn_student
#
#
# if __name__ == '__main__':
#     conv_model = get_self_base_fsrcnn_student(2, 1)
#     adder_model = get_self_fsrcnn_student(2, 1)
#
#     input = torch.randn(1, 1, 32, 32)
#     macs, params = profile(conv_model, inputs=(input,))
#     macs, params = clever_format([macs, params], "%.3f")
#     print(macs)
#     print('-'*50)
#     print(params)
#     print('-'*50)
#     print('#'*50)
#
#     macs, params = profile(adder_model, inputs=(input,))
#     macs, params = clever_format([macs, params], "%.3f")
#     print(macs)
#     print('-' * 50)
#     print(params)
#     print('-' * 50)
#     print('#' * 50)










from torchstat import stat

from models.self_adder_fsrcnn import get_self_fsrcnn_student, get_self_base_fsrcnn_student

if __name__ == '__main__':
    conv_model = get_self_base_fsrcnn_student(2, 1)
    adder_model = get_self_fsrcnn_student(2, 1)
    stat(conv_model, (1, 32, 32))
    print('-'*50)
    stat(adder_model, (1, 32, 32))
