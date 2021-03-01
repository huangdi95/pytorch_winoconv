#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Fri 26 Feb 2021 10:27:01 AM CST
#########################################################################
import torch
from torch import nn
import torch.nn.functional as F
import dwm
import time

#model = nn.Sequential(
#    nn.Conv2d(3,20,5),
#    nn.ReLU(),
#    nn.Conv2d(20,64,5),
#    nn.ReLU(),
#).cuda()
class Dwm(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.Tensor())
        self.register_buffer('tmp_input_buffer', torch.Tensor(kernel_size, B, nD, nH, nW, C).float())
        self.register_buffer('tmp_weight_buffer', torch.Tensor(kernel_size, C, K).float())
        self.register_buffer('tmp_product_buffer', torch.Tensor(kernel_size * nD * nH * nW * B * K).float())
        self.register_buffer('tmp_ptr_buffer', torch.Tensor(3 * kernel_size).long())

    def forward(self, x, w):
        y = dwm.dwm3d(x, w, None, (1, 1, 1))
        return y

x = torch.rand(1, 32, 8, 28, 28).cuda()
w = torch.rand(32, 32, 3, 3, 3).cuda()


y = F.conv3d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)
y = F.conv3d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)
t = time.perf_counter()
y = F.conv3d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)
torch.cuda.synchronize()

Time = time.perf_counter() - t
print(Time)



print('--- warm up ----')
y = dwm.dwm3d(x, w, None, (1, 1, 1))
y = dwm.dwm3d(x, w, None, (1, 1, 1))
print('----------------')
t = time.perf_counter()
y = dwm.dwm3d(x, w, None, (1, 1, 1))
torch.cuda.synchronize()

Time = time.perf_counter() - t
print(Time)
