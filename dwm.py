import torch
from MakePytorchBackend import DWM, DWM2D
import time

def dwm3d(input, kernel, bias=None, stride=(1, 1, 1), padding=0, groups=1):
    if type(padding) == type((1, 2)):
        pad = (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
        input = torch.nn.functional.pad(input, pad)
    #(N, H, W, C)
#    input = input.cuda().float()
#    kernel = kernel.cuda().float()
    t = time.perf_counter()
    out= dwm3d_(input, kernel)
    Time = time.perf_counter() - t
    print(Time)
#    print(kernel.size)
    if not bias is None:
        out = out.permute(0, 2, 3, 4, 1)
#        out = out.half() + bias.half() 
        out = out + bias.type_as(out) 
        out = out.permute(0, 4, 1, 2, 3)
#    out = out.float()
    return out

def dwm2d(input, kernel, bias=None, stride=(1, 1), padding=0, groups=1):
    if type(padding) == type((1, 2)):
        pad = (padding[1], padding[1], padding[0], padding[0])
        input = torch.nn.functional.pad(input, pad)
    #(N, H, W, C)
#    input = input.cuda().float()
#    kernel = kernel.cuda().float()
    out= dwm2d_(input, kernel)
#    print(kernel.size)
    if bias is not None:
        out = out.permute(0, 2, 3, 1)
#        out = out.half() + bias.half() 
        out = out + bias.type_as(out) 
        out = out.permute(0, 3, 1, 2)
#    out = out.float()
    return out


def dwm3d_(x, w, stride=1):
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    B = int(list(x.size())[0])
    C = int(list(x.size())[1])
    D = int(list(x.size())[2])
    H = int(list(x.size())[3])
    W = int(list(x.size())[4])
    K = int(list(w.size())[0])
    w_D = int(list(w.size())[2])
    w_H = int(list(w.size())[3])
    w_W = int(list(w.size())[4])
    kernel_size = int((w_D + 1 + (w_D - 1) / 3) * (w_H + 1 + (w_H - 1) / 3) * (w_W + 1 + (w_W - 1) / 3))
    output_D = int((D  - w_D) / stride + 1)
    output_H = int((H  - w_H) / stride + 1)
    output_W = int((W  - w_W) / stride + 1)
    nD = int((output_D + 1) / 2)
    nH = int((output_H + 1) / 2)
    nW = int((output_W + 1) / 2)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    tmp_input_buffer = torch.empty(kernel_size, B, nD, nH, nW, C, dtype=x.dtype, device=x.device)
    tmp_weight_buffer = torch.empty(kernel_size, C, K, dtype=x.dtype, device=x.device)
    tmp_product_buffer = torch.empty(kernel_size * nD * nH * nW * B * K, dtype=x.dtype, device=x.device)
    tmp_ptr_buffer = torch.empty(3 * kernel_size, dtype=torch.long, device=x.device)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

#    out = torch.Tensor(B, output_D, output_H, output_W, K).cuda().type_as(x)
    out = torch.zeros(B, output_D, output_H, output_W, K, dtype=x.dtype, device=x.device)
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    x = x.permute(0, 2, 3, 4, 1).contiguous()
    w = w.permute(2, 3, 4, 1, 0).contiguous()
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    DWM(x, w, out, stride,
        tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, tmp_ptr_buffer)
    torch.cuda.synchronize()
    t6 = time.perf_counter()

    del tmp_input_buffer
    del tmp_weight_buffer
    del tmp_product_buffer
    del tmp_ptr_buffer
    torch.cuda.synchronize()
    t7 = time.perf_counter()

    out = out.permute(0, 4, 1, 2, 3).contiguous()
    torch.cuda.synchronize()
    t8 = time.perf_counter()
    print('init var:', t2 - t1)
    print('init tmp:', t3 - t2)
    print('init out:', t4 - t3)
    print('permute: ', t5 - t4)
    print('dwm:     ', t6 - t5)
    print('del buff:', t7 - t6)
    print('pm o:    ', t8 - t7)
    return out

def dwm2d_(x, w, stride=1):
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    B = int(list(x.size())[0])
    C = int(list(x.size())[1])
    H = int(list(x.size())[2])
    W = int(list(x.size())[3])
    K = int(list(w.size())[0])
    w_H = int(list(w.size())[2])
    w_W = int(list(w.size())[3])
    kernel_size = int((w_H + 1 + int((w_H - 1) / 3)) * (w_W + 1 + int((w_W - 1) / 3)) * 2)
    output_H = int((H  - w_H) / stride + 1)
    output_W = int((W  - w_W) / stride + 1)
    nH = int((output_H + 1) / 2)
    nW = int((output_W + 1) / 2)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    tmp_input_buffer = torch.empty(kernel_size, B, nH, nW, C, dtype=x.dtype, device=x.device)
    tmp_weight_buffer = torch.empty(kernel_size, C, K, dtype=x.dtype, device=x.device)
    tmp_product_buffer = torch.empty(kernel_size * nH * nW * B * K, dtype=x.dtype, device=x.device)
    tmp_ptr_buffer = torch.empty(3 * kernel_size, dtype=torch.long, device=x.device)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    out = torch.zeros(B, output_H, output_W, K, dtype=x.dtype, device=x.device)
    t4 = time.perf_counter()

    x = x.permute(0, 2, 3, 1).contiguous()
    w = w.permute(2, 3, 1, 0).contiguous()
#    print(w.shape)
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    DWM2D(x, w, out, stride,
        tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, tmp_ptr_buffer)
    torch.cuda.synchronize()
    t6 = time.perf_counter()

    del tmp_input_buffer
    del tmp_weight_buffer
    del tmp_product_buffer
    del tmp_ptr_buffer

    torch.cuda.synchronize()
    t7 = time.perf_counter()
    out = out.permute(0, 3, 1, 2).contiguous()
    torch.cuda.synchronize()
    t8 = time.perf_counter()
    print('init var:', t2 - t1)
    print('init tmp:', t3 - t2)
    print('init out:', t4 - t3)
    print('permute: ', t5 - t4)
    print('dwm:     ', t6 - t5)
    print('del buff:', t7 - t6)
    print('pm o:    ', t8 - t7)
    return out
