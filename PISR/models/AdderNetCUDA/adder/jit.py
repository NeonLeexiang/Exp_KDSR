# from torch.utils.cpp_extension import load
#
# conv_cuda = load(
#     'adder_cuda', ['adder_cuda.cpp', 'adder_cuda_kernel.cu'], verbose=True)
import torch
import adder_cuda
help(adder_cuda)
