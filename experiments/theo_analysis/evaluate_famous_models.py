import torch
from torch import nn
from thop.profile import profile
import models

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") # and "inception" in name
                     and callable(models.__dict__[name]))

print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


size_dict = {
    'vox2vox': (1, 3, 16, 112, 112),
    'voxnet': (1, 1, 32, 32, 32),
    'shapenet': (1, 1, 30, 30, 30),
    'vnet': (1, 1, 64, 128, 128),
    'spynet': (1, 3, 448, 1024),
    'inpaintingnet': (1, 4, 32, 128, 128),
    'toflow_denoise': (1, 7, 3, 256, 448),
    'toflow_sr': (1, 7, 3, 256, 448),
    'toflow_interp': (1, 2, 3, 256, 448),
    'slowfast_resnet50': (1, 3, 64, 224, 224),
    'slowfast_resnet101': (1, 3, 64, 224, 224),
    'slowfast_resnet152': (1, 3, 64, 224, 224),
    'slowfast_resnet200': (1, 3, 64, 224, 224),
    'completionnet': (1, 32, 3, 128, 128),
    'videoinpaintingmodel': (1, 32, 3, 128, 128),
}
for name in model_names:
    if name == 'test': continue
    model = models.__dict__[name]().to(device)
    dsize = size_dict[name]
    #    if "inception" in name:
    #        dsize = (1, 3, 299, 299)
    inputs = torch.rand(dsize).to(device)
    if name in ['spynet']:
        inputs = (inputs, inputs)
    elif name in ['completionnet']:
        mask = torch.rand((1, 32, 1, 128, 128)).to(device)
        inputs = (inputs, mask)
    elif name in ['videoinpaintingmodel']:
        mask = torch.rand((1, 32, 1, 128, 128)).to(device)
        guidances = torch.rand((1, 32, 1, 128, 128)).to(device)
        inputs = (inputs, mask, guidances)
    else:
        inputs = (inputs, )
    total_ops, total_params = profile(model, inputs, verbose=False)
    print("%s | %.2f | %.2f" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))
