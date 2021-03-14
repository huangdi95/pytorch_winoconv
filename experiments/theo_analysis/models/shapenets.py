import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ['ShapeNet', 'shapenet']

class ShapeNet(nn.Module):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(ShapeNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = nn.Sequential(
                    nn.Conv3d(1, 48, 6, stride=2),
                    nn.Conv3d(48, 160, 5, stride=2),
                    nn.Conv3d(160, 512, 4, stride=1),
                    )
        self.mlp = nn.Sequential(
                   nn.Linear(8*512, 1200),
                   nn.Linear(1200, 4000),
                   nn.Linear(4000, self.n_classes),
                   )

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

def shapenet():
    return ShapeNet()
