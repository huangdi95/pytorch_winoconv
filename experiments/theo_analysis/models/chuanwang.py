#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Wed 02 Dec 2020 11:48:05 AM CST
#########################################################################
import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ['InpaintingNet', 'inpaintingnet']

class InpaintingNet(nn.Module):
    def __init__(self):
        super(InpaintingNet, self).__init__()
        self.feat1_1 = nn.Sequential(
                       nn.Conv3d(4, 16, 5, padding=2),
                       nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 0),
                       nn.Conv3d(16, 32, 3, stride=(1, 2, 2)),
                       )
        self.feat1_2 = nn.Conv3d(32, 64, 3, padding=1)
        self.feat1_3 = nn.Sequential(
                       nn.ConstantPad3d((2, 1, 2, 1, 1, 1), 0),
                       nn.Conv3d(64, 128, 3, stride=(1, 2, 2)),
                       )
        self.feat2_3 = nn.Sequential(
                       nn.Conv3d(128, 256, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
                       nn.Conv3d(256, 256, 3, padding=(1, 4, 4), dilation=(1, 4, 4)),
                       nn.Conv3d(256, 256, 3, padding=(1, 8, 8), dilation=(1, 8, 8)),
                       nn.Conv3d(256, 128, 3, padding=1),
                       )
        self.feat2_2 = nn.Sequential(
                       nn.ConstantPad3d((-1, -1, -1, -1, -2, -1), 0),
                       nn.ConvTranspose3d(128, 64, 4, stride=(1, 2, 2))
                       )
        self.feat2_1 = nn.Conv3d(64, 32, 3, padding=1)
        self.feat2_0 = nn.Sequential(
                       nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
                       nn.Conv3d(16, 3, 3, padding=1),
                       )

        self.feat1 = nn.Conv2d(3, 128, 3, padding=1)
        self.feat2 = nn.Conv2d(3, 128, 3, padding=1)

        self.feat3_1 = nn.Conv2d(4, 64, 5, padding=2)
        self.feat3_2 = nn.Sequential(
                       nn.ConstantPad2d((1, 1, 1, 1), 0),
                       nn.Conv2d(64, 128, 3, stride=(2, 2)),
                       )
        self.feat3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.feat3_4 = nn.Sequential(
                       nn.ConstantPad2d((1, 1, 1, 1), 0),
                       nn.Conv2d(128, 256, 3, stride=(2, 2)),
                       )
        self.feat3_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.feat3_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.feat3_7 = nn.Conv2d(256, 256, 3, padding=(2, 2), dilation=(2, 2))
        self.feat4_7 = nn.Sequential(
                       nn.Conv2d(256, 256, 3, padding=(4, 4), dilation=(4, 4)),
                       nn.Conv2d(256, 256, 3, padding=(8, 8), dilation=(8, 8)),
                       )
        self.feat4_6 = nn.Conv2d(256, 256, 3, padding=(16, 16), dilation=(16, 16))
        self.feat4_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.feat4_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.feat4_3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.feat4_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.feat4_1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.feat4_0 = nn.Sequential(
                       nn.Conv2d(64, 32, 3, padding=1),
                       nn.Conv2d(32, 3, 3, padding=1),
                       )

    def forward(self, v):
        v_downsample = v[:,:,:,:64,:64]
        v1_1 = self.feat1_1(v_downsample)
        v1_2 = self.feat1_2(v1_1)
        v1_3 = self.feat1_3(v1_2)
        v2_3 = self.feat2_3(v1_3)
        v2_3 = v2_3 + v1_3
        v2_2 = self.feat2_2(v2_3)
        v2_2 = v2_2 + v1_2
        v2_1 = self.feat2_1(v2_2)
        v2_1 = v2_1 + v1_1
        v2_0 = self.feat2_0(v2_1)

        output = []
        for i in range(32):
            f = v[:,:,i,:,:]
            v_1 = self.feat1(v2_0[:,:,i,:,:])
            v_2 = self.feat2(v2_0[:,:,i,:,:])
            f3_1 = self.feat3_1(f) 
            f3_2 = self.feat3_2(f3_1)
            f3_2 = f3_2 + v_1
            f3_3 = self.feat3_3(f3_2)
            f3_4 = self.feat3_4(f3_3)
            f3_5 = self.feat3_5(f3_4)
            f3_6 = self.feat3_6(f3_5)
            f3_7 = self.feat3_7(f3_6)
            f4_7 = self.feat3_7(f3_7)
            f4_7 = f4_7 + f3_7
            f4_6 = self.feat4_6(f4_7)
            f4_6 = f4_6 + f3_6
            f4_5 = self.feat4_5(f4_6)
            f4_5 = f4_5 + f3_5
            f4_4 = self.feat4_4(f4_5)
            f4_4 = f4_4 + f3_4
            f4_3 = self.feat4_3(f4_4)
            f4_3 = f4_3 + f3_3
            f4_2 = self.feat4_2(f4_3)
            f4_2 = f4_2 + f3_2
            f4_2 = f4_2 + v_2
            f4_1 = self.feat4_1(f4_2)
            f4_1 = f4_1 + f3_1
            f4_0 = self.feat4_0(f4_1)
            output.append(f4_0)

        output = torch.cat(output, dim=2)

        return output

def inpaintingnet():
    return InpaintingNet()
