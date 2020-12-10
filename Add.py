import torch

from MakePytorchBackend import AddGPU


def add_gpu(a, b):
    assert a.numel() == b.numel()

    c = a.new()
    AddGPU(a, b, c)
    return c
