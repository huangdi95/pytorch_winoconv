#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Tue 01 Dec 2020 10:50:50 AM CST
#########################################################################
import torch
import torch.nn as nn

#name = 'test'
__all__ = ['Test', 'test']
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(3, 3, 3),
                    nn.Conv2d(3, 3,3))
   
    def forward(self, x):
        return self.conv(x) 

def test():
    return Test()
