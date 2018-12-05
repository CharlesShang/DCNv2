#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dcn_v2 import DCNv2
from dcn_v2_func import DCNv2Function

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3

def conv_identify(weight, bias):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, y, x] = 1.0

def check_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
        kernel_size=(kH, kW),
        stride=(1, 1),
        padding=(1, 1),
        bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
        kernel_size=(kH, kW),
        stride=(1, 1),
        padding=(1, 1),
        bias=True).cuda()

    dcn_v2 = DCNv2(inC, outC, (kH, kW),
                   stride=1, padding=1, dilation=1,
                   deformable_groups=deformable_groups).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn_v2.weight, dcn_v2.bias)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn_v2(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('Zero offset passed')
    else:
        print('Zero offset failed')

def check_gradient_double():

    input = torch.randn(N, inC, inH, inW, dtype=torch.float64).cuda()
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW, dtype=torch.float64).cuda()
    offset.data.zero_()
    offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW, dtype=torch.float64).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, inC, kH, kW, dtype=torch.float64).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC, dtype=torch.float64).cuda()
    bias.requires_grad = True

    func = DCNv2Function(stride=1, padding=1, dilation=1, deformable_groups=deformable_groups)

    print(gradcheck(func, (input, offset, mask, weight, bias), eps=1e-6, atol=1e-5, rtol=1e-3))

def check_gradient():

    input = torch.randn(N, inC, inH, inW).cuda()
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda()
    offset.data.zero_()
    offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, inC, kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    func = DCNv2Function(stride=1, padding=1, dilation=1, deformable_groups=deformable_groups)

    print(gradcheck(func, (input, offset, mask, weight, bias), eps=1e-3, atol=1e-3, rtol=1e-2))

def example():
    from dcn_v2 import DCN
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)

if __name__ == '__main__':

    example()

    # zero offset check
    if inC == outC:
        check_zero_offset()

    # gradient check
    try:
        check_gradient_double()
    except TypeError:
        print('''You can swith to double precision in dcn_v2_func.py by (un)commenting these two lines: 
                 from _ext import dcn_v2 as _backend
                 from _ext import dcn_v2_double as _backend''')
        print('Your tensor may not be **double** type')
        print('Switching to **float** type')

        check_gradient()
    finally:
        print('Note: backward is not reentrant error may not be a serious problem, '
              'since the max error is less than 1e-7\n'
              'Still looking for what trigger this problem')

    example()