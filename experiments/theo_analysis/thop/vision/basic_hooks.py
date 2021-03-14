import argparse
import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
import numpy as np

multiply_adds = 1

Wino = False
Wino2 = False 
DWM = False 

def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = total_params


def zero_ops(m, x, y):
    m.total_ops += torch.DoubleTensor([int(0)])


def count_deconvNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    total_ops = wino_count(total_ops, m.weight.size()[2:], np.ones(len(m.stride)))

    m.total_ops += torch.DoubleTensor([int(total_ops)])

def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    total_ops = wino_count(total_ops, m.weight.size()[2:], m.stride)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_convNd_ver2(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
    # Cout x Cin x Kw x Kh
    kernel_ops = m.weight.nelement()
#    if m.bias is not None:
#        # Cout x 1
#        kernel_ops += + m.bias.nelement()
    # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    m.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.DoubleTensor([int(nelements)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor([*(y.shape[2:])])
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops = total_mul * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])

def wino_count(total_ops, size, stride):
    if Wino:
        if list(size) == [3, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 3.38
        elif list(size) == [1, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [3, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [3, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [3, 1, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [3, 3] and list(stride) == [1, 1]:
            total_ops /= 2.25
        elif list(size) == [1, 3] and list(stride) == [1, 1]:
            total_ops /= 1.5
        elif list(size) == [3, 1] and list(stride) == [1, 1]:
            total_ops /= 1.5
    if Wino2:
        if list(size) == [3, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 3.38
        elif list(size) == [1, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [3, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [3, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [3, 1, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5

#        elif list(size) == [1, 4, 4] and list(stride) == [1, 1, 1]:
#            total_ops /= 1.77
        elif list(size) == [1, 5, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 0.33
        elif list(size) == [1, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 0.10
        elif list(size) == [1, 9, 9] and list(stride) == [1, 1, 1]:
            total_ops /= 0.05
        elif list(size) == [1, 11, 11] and list(stride) == [1, 1, 1]:
            total_ops /= 0.02

        elif list(size) == [4, 4, 4] and list(stride) == [1, 1, 1]:
            total_ops /= 0.14
        elif list(size) == [5, 5, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 0.13
        elif list(size) == [6, 6, 6] and list(stride) == [1, 1, 1]:
            total_ops /= 0.13
        elif list(size) == [7, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 0.12
        elif list(size) == [3, 5, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 0.14
        elif list(size) == [3, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 0.13
        elif list(size) == [5, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 0.12
        elif list(size) == [8, 4, 4] and list(stride) == [1, 1, 1]:
            total_ops /= 0.12
        elif list(size) == [10, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 0.11 

        elif list(size) == [3, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
#        elif list(size) == [4, 4, 1] and list(stride) == [1, 1, 1]:
#            total_ops /= 1.77
        elif list(size) == [5, 5, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 0.33
        elif list(size) == [7, 7, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 0.10
        elif list(size) == [9, 9, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 0.05
        elif list(size) == [11, 11, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 0.02

        elif list(size) == [3, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
#        elif list(size) == [4, 1, 4] and list(stride) == [1, 1, 1]:
#            total_ops /= 1.77
        elif list(size) == [5, 1, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 0.33
        elif list(size) == [7, 1, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 0.10
        elif list(size) == [9, 1, 9] and list(stride) == [1, 1, 1]:
            total_ops /= 0.05
        elif list(size) == [11, 1, 11] and list(stride) == [1, 1, 1]:
            total_ops /= 0.02

        elif list(size) == [3, 1, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 3] and list(stride) == [1, 1]:
            total_ops /= 1.5
        elif list(size) == [3, 1] and list(stride) == [1, 1]:
            total_ops /= 1.5

        elif list(size) == [3, 3] and list(stride) == [1, 1]:
            total_ops /= 2.25
#        elif list(size) == [4, 4] and list(stride) == [1, 1]:
#            total_ops /= 1.77
        elif list(size) == [5, 5] and list(stride) == [1, 1]:
            total_ops /= 0.33
        elif list(size) == [7, 7] and list(stride) == [1, 1]:
            total_ops /= 0.10
        elif list(size) == [9, 9] and list(stride) == [1, 1]:
            total_ops /= 0.05
        elif list(size) == [11, 11] and list(stride) == [1, 1]:
            total_ops /= 0.02

    elif DWM:
        if list(size) == [3, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 3.38
        elif list(size) == [4, 4, 4] and list(stride) == [1, 1, 1]:
            total_ops /= 2.37
        elif list(size) == [5, 5, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 2.92
        elif list(size) == [6, 6, 6] and list(stride) == [1, 1, 1]:
            total_ops /= 3.38
        elif list(size) == [7, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 2.74
        elif list(size) == [3, 5, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 3.06
        elif list(size) == [3, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 2.94
        elif list(size) == [5, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 3.01
        elif list(size) == [8, 4, 4] and list(stride) == [1, 1, 1]:
            total_ops /= 2.37
        elif list(size) == [10, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 3.21 

        elif list(size) == [3, 3, 3] and list(stride) == [2, 2, 2]:
            total_ops /= 1.73
        elif list(size) == [4, 4, 4] and list(stride) == [2, 2, 2]:
            total_ops /= 2.37
        elif list(size) == [5, 5, 5] and list(stride) == [2, 2, 2]:
            total_ops /= 2.92
        elif list(size) == [6, 6, 6] and list(stride) == [2, 2, 2]:
            total_ops /= 3.38
        elif list(size) == [7, 7, 7] and list(stride) == [2, 2, 2]:
            total_ops /= 2.74
        elif list(size) == [3, 5, 5] and list(stride) == [2, 2, 2]:
            total_ops /= 2.45
        elif list(size) == [3, 7, 7] and list(stride) == [2, 2, 2]:
            total_ops /= 2.56
        elif list(size) == [5, 7, 7] and list(stride) == [2, 2, 2]:
            total_ops /= 3.06
        elif list(size) == [8, 4, 4] and list(stride) == [2, 2, 2]:
            total_ops /= 2.37
        elif list(size) == [10, 3, 3] and list(stride) == [2, 2, 2]:
            total_ops /= 2.06 
        elif list(size) == [8, 4, 4] and list(stride) == [2, 2, 2]:
            total_ops /= 2.06 


        elif list(size) == [1, 3, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [1, 4, 4] and list(stride) == [1, 1, 1]:
            total_ops /= 1.77
        elif list(size) == [1, 5, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 2.04
        elif list(size) == [1, 7, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 1.96
        elif list(size) == [1, 9, 9] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [1, 11, 11] and list(stride) == [1, 1, 1]:
            total_ops /= 2.15

        elif list(size) == [3, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [4, 4, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.77
        elif list(size) == [5, 5, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 2.04
        elif list(size) == [7, 7, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.96
        elif list(size) == [9, 9, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [11, 11, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 2.15

        elif list(size) == [3, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [4, 1, 4] and list(stride) == [1, 1, 1]:
            total_ops /= 1.77
        elif list(size) == [5, 1, 5] and list(stride) == [1, 1, 1]:
            total_ops /= 2.04
        elif list(size) == [7, 1, 7] and list(stride) == [1, 1, 1]:
            total_ops /= 1.96
        elif list(size) == [9, 1, 9] and list(stride) == [1, 1, 1]:
            total_ops /= 2.25
        elif list(size) == [11, 1, 11] and list(stride) == [1, 1, 1]:
            total_ops /= 2.15

        elif list(size) == [3, 1, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 3, 1] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 1, 3] and list(stride) == [1, 1, 1]:
            total_ops /= 1.5
        elif list(size) == [1, 3] and list(stride) == [1, 1]:
            total_ops /= 1.5
        elif list(size) == [3, 1] and list(stride) == [1, 1]:
            total_ops /= 1.5

        elif list(size) == [1, 3, 3] and list(stride) == [1, 2, 2]:
            total_ops /= 1.44
        elif list(size) == [1, 5, 5] and list(stride) == [1, 2, 2]:
            total_ops /= 2.04
        elif list(size) == [1, 7, 7] and list(stride) == [1, 2, 2]:
            total_ops /= 1.96
        elif list(size) == [1, 9, 9] and list(stride) == [1, 2, 2]:
            total_ops /= 1.92
        elif list(size) == [1, 11, 11] and list(stride) == [1, 2, 2]:
            total_ops /= 2.15


        elif list(size) == [3, 3] and list(stride) == [1, 1]:
            total_ops /= 2.25
        elif list(size) == [4, 4] and list(stride) == [1, 1]:
            total_ops /= 1.77
        elif list(size) == [5, 5] and list(stride) == [1, 1]:
            total_ops /= 2.04
        elif list(size) == [7, 7] and list(stride) == [1, 1]:
            total_ops /= 1.96
        elif list(size) == [9, 9] and list(stride) == [1, 1]:
            total_ops /= 2.25
        elif list(size) == [11, 11] and list(stride) == [1, 1]:
            total_ops /= 2.15

        elif list(size) == [3, 3] and list(stride) == [2, 2]:
            total_ops /= 1.44
        elif list(size) == [5, 5] and list(stride) == [2, 2]:
            total_ops /= 2.04
        elif list(size) == [7, 7] and list(stride) == [2, 2]:
            total_ops /= 1.96
        elif list(size) == [9, 9] and list(stride) == [2, 2]:
            total_ops /= 1.92
        elif list(size) == [11, 11] and list(stride) == [2, 2]:
            total_ops /= 2.15


        elif list(size) == [5, 7, 7] and list(stride) == [1, 2, 2]:
            total_ops /= 2.8


        else:
            print('Error! Unknown kernel size & stride:', list(size), list(stride))

    return total_ops
