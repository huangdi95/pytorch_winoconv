#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Tue 01 Dec 2020 10:50:50 AM CST
#########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['V2V', 'vox2vox']
class V2V(nn.Module):
    def __init__(self):
        super(V2V, self).__init__()
#        self.c3d = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=0)
        self.conv3b = nn.Sequential(
                      nn.Conv3d(3, 64, 3, padding=1),
                      nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
                      nn.Conv3d(64, 128, 3, padding=1),
                      nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
                      nn.Conv3d(128, 256, 3, padding=1),
                      nn.Conv3d(256, 256, 3, padding=1),
                      )
        self.conv4b = nn.Sequential( 
                      nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
                      nn.Conv3d(256, 512, 3, padding=1),
                      nn.Conv3d(512, 512, 3, padding=1),
                      )
        self.conv5b = nn.Sequential( 
                      nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
                      nn.Conv3d(512, 512, 3, padding=1),
                      nn.Conv3d(512, 512, 3, padding=1),
                      )
        self.conv3c = nn.Conv3d(256, 64, 3, padding=1)
        self.conv4c = nn.Conv3d(512, 64, 3, padding=1)
        self.deconv3 = nn.ConvTranspose3d(128, 64, (8, 4, 4), stride=(4, 2, 2), padding=(2, 1, 1))
        self.deconv4 = nn.ConvTranspose3d(128, 64, 4, stride=(2, 2, 2), padding=(1, 1, 1))
        self.deconv5 = nn.ConvTranspose3d(512, 64, 4, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv_pre = nn.Conv3d(64, 8, 3, padding=1)
        
    def forward(self, x):
        c3b = self.conv3b(x)
        c4b = self.conv4b(c3b)
        c5b = self.conv5b(c4b)
        c3c = self.conv3c(c3b)
        c4c = self.conv4c(c4b)
        de5 = self.deconv5(c5b)
        de4 = self.deconv4(torch.cat([c4c, de5], dim=1))
        de3 = self.deconv3(torch.cat([c3c, de4], dim=1))
        out = self.conv_pre(de3)
        return out

def vox2vox():
    return V2V()
