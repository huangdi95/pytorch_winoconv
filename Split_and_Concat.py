import torch
from MakePytorchBackend import WinoSplit, WinoConcat 

def wino_split(a, w_shape, output_block_size):
    if len(w_shape) == 5:
        B = list(a.size())[0]
        C = list(a.size())[1]
        D = list(a.size())[2]
        H = list(a.size())[3]
        W = list(a.size())[4]
        w_D = w_shape[2]
        w_H = w_shape[3]
        w_W = w_shape[4]
        output_block_D = output_block_H = output_block_W = int(output_block_size)
        block_D = w_D + output_block_D - 1
        block_H = w_H + output_block_H - 1
        block_W = w_W + output_block_W - 1
        pad_d = 0
        pad_h = 0
        pad_w = 0
        nD = int((D + 1 + 2 * pad_d - block_D) / output_block_D + 1)
        nH = int((H + 1 + 2 * pad_h - block_H) / output_block_H + 1)
        nW = int((W + 1 + 2 * pad_w - block_W) / output_block_W + 1)
        out = torch.zeros(B*nD*nH*nW, C, block_D, block_H, block_W).cuda()

        a = a.permute([0, 2, 3, 4, 1]).contiguous()
        WinoSplit(a, out, block_D, block_H, block_W, output_block_size, nD, nH, nW, False)
    else:
        B = list(a.size())[0]
        C = list(a.size())[1]
        H = list(a.size())[2]
        W = list(a.size())[3]
        w_H = w_shape[2]
        w_W = w_shape[3]
        output_block_H = output_block_W = int(output_block_size)
        block_H = w_H + output_block_H - 1
        block_W = w_W + output_block_W - 1
        pad_h = 0
        pad_w = 0
        nH = int((H + 1 + 2 * pad_h - block_H) / output_block_H + 1)
        nW = int((W + 1 + 2 * pad_w - block_W) / output_block_W + 1)
        out = torch.zeros(B*nH*nW, C, block_H, block_W).cuda()

        nD = 0 
        block_D = 0

        a = a.permute([0, 2, 3, 1]).contiguous()
        WinoSplit(a, out, block_D, block_H, block_W, output_block_size, nD, nH, nW, True)
    return out

def wino_concat(a, output_size, output_block_size):
    if len(output_size) == 3:
        N = list(a.size())[0]
        K = list(a.size())[1]
        output_D = int(output_size[0])
        output_H = int(output_size[1])
        output_W = int(output_size[2])
        output_block_D = output_block_H = output_block_W = int(output_block_size)
        nD = int((output_D+output_block_D-1)/output_block_D)
        nH = int((output_H+output_block_H-1)/output_block_H)
        nW = int((output_W+output_block_W-1)/output_block_W)
        B = int(N / (nD * nH * nW))
        out = torch.zeros(B, output_D, output_H, output_W, K).cuda()

        a = a.contiguous()
        WinoConcat(a, out, output_block_size, False)
        out = out.permute([0, 4, 1, 2, 3]).contiguous()
    else:
        N = list(a.size())[0]
        K = list(a.size())[1]
        output_H = int(output_size[0])
        output_W = int(output_size[1])
        output_block_H = output_block_W = int(output_block_size)
        nH = int((output_H+output_block_H-1)/output_block_H)
        nW = int((output_W+output_block_W-1)/output_block_W)
        B = int(N / (nH * nW))
        out = torch.zeros(B, output_H, output_W, K).cuda()

        a = a.contiguous()
        WinoConcat(a, out, output_block_size, True)
        out = out.permute([0, 3, 1, 2]).contiguous()
    return out
