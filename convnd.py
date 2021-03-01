#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Fri 11 Dec 2020 05:29:21 PM CST
#########################################################################
import torch
import torch.nn.functional as F
import numpy as np
import matrix_convnd
import dwm_convnd


def conv3d(x, w):
    n = 3
    out3 = F.conv3d(x, w)
    out_all = []
    for b in range(x.shape[0]):
        out_b = []
        for k in range(w.shape[0]):
            x_tmp = x[b]
            w_tmp = w[k]
            w_size = w_tmp.shape[-1]
            x_size = x_tmp.shape[-1]
            x_tmp = x_tmp.view(-1, 1, *list(x_tmp.shape[-(n-1):]))
            w_tmp = w_tmp.view(-1, 1, *list(w_tmp.shape[-(n-1):]))
            out = F.conv2d(x_tmp, w_tmp)
            index = torch.Tensor(range(w_size)).long().cuda()
            index = index.view(-1, 1)
            index = torch.cat([index]*np.prod(out.shape[2:]), -1)

            index = index.view([w_size, 1, *list(out.shape[2:])])
            out1 = out[:-1].gather(1, index)
            out2 = out[1:].gather(1, index)
            out1 = out1.sum(0)
            out2 = out2.sum(0)
            out = torch.cat([out1, out2], 0)
            out_b.append(out.view(1, *list(out.shape)))
        out_all.append(torch.cat(out_b, dim=0).view(1, -1, *list(out.shape)))
    out = torch.cat(out_all, dim=0)
#    print((out-out3) / out3)
    return out

def conv4d(x, w):
    n = 4
    out_all = []
    for b in range(x.shape[0]):
        out_b = []
        for k in range(w.shape[0]):
            x_tmp = x[b]
            w_tmp = w[k]
            w_size = w_tmp.shape[-1]
            x_size = x_tmp.shape[-1]
            x_tmp = x_tmp.view(-1, 1, *list(x_tmp.shape[-(n-1):]))
            w_tmp = w_tmp.view(-1, 1, *list(w_tmp.shape[-(n-1):]))
            out = F.conv3d(x_tmp, w_tmp)
            index = torch.Tensor(range(w_size)).long().cuda()
            index = index.view(-1, 1)
            index = torch.cat([index]*np.prod(out.shape[2:]), -1)

            index = index.view([w_size, 1, *list(out.shape[2:])])
            out1 = out[:-1].gather(1, index)
            out2 = out[1:].gather(1, index)
            out1 = out1.sum(0)
            out2 = out2.sum(0)
            out = torch.cat([out1, out2], 0)
            out_b.append(out.view(1, *list(out.shape)))
        out_all.append(torch.cat(out_b, dim=0).view(1, -1, *list(out.shape)))
    out = torch.cat(out_all, dim=0)
    return out

def conv5d(x, w):
    n = 5
    out_all = []
    for b in range(x.shape[0]):
        out_b = []
        for k in range(w.shape[0]):
            x_tmp = x[b]
            w_tmp = w[k]
            w_size = w_tmp.shape[-1]
            x_size = x_tmp.shape[-1]
            x_tmp = x_tmp.view(-1, 1, *list(x_tmp.shape[-(n-1):]))
            w_tmp = w_tmp.view(-1, 1, *list(w_tmp.shape[-(n-1):]))
            out = conv4d(x_tmp, w_tmp)
            index = torch.Tensor(range(w_size)).long().cuda()
            index = index.view(-1, 1)
            index = torch.cat([index]*np.prod(out.shape[2:]), -1)

            index = index.view([w_size, 1, *list(out.shape[2:])])
            out1 = out[:-1].gather(1, index)
            out2 = out[1:].gather(1, index)
            out1 = out1.sum(0)
            out2 = out2.sum(0)
            out = torch.cat([out1, out2], 0)
            out_b.append(out.view(1, *list(out.shape)))
        out_all.append(torch.cat(out_b, dim=0).view(1, -1, *list(out.shape)))
    out = torch.cat(out_all, dim=0)
    return out

def conv6d(x, w):
    n = 6
    out_all = []
    for b in range(x.shape[0]):
        out_b = []
        for k in range(w.shape[0]):
            x_tmp = x[b]
            w_tmp = w[k]
            w_size = w_tmp.shape[-1]
            x_size = x_tmp.shape[-1]
            x_tmp = x_tmp.view(-1, 1, *list(x_tmp.shape[-(n-1):]))
            w_tmp = w_tmp.view(-1, 1, *list(w_tmp.shape[-(n-1):]))
            out = conv5d(x_tmp, w_tmp)
            index = torch.Tensor(range(w_size)).long().cuda()
            index = index.view(-1, 1)
            index = torch.cat([index]*np.prod(out.shape[2:]), -1)

            index = index.view([w_size, 1, *list(out.shape[2:])])
            out1 = out[:-1].gather(1, index)
            out2 = out[1:].gather(1, index)
            out1 = out1.sum(0)
            out2 = out2.sum(0)
            out = torch.cat([out1, out2], 0)
            out_b.append(out.view(1, *list(out.shape)))
        out_all.append(torch.cat(out_b, dim=0).view(1, -1, *list(out.shape)))
    out = torch.cat(out_all, dim=0)
    return out

def conv7d(x, w):
    n = 7
    out_all = []
    for b in range(x.shape[0]):
        out_b = []
        for k in range(w.shape[0]):
            x_tmp = x[b]
            w_tmp = w[k]
            w_size = w_tmp.shape[-1]
            x_size = x_tmp.shape[-1]
            x_tmp = x_tmp.view(-1, 1, *list(x_tmp.shape[-(n-1):]))
            w_tmp = w_tmp.view(-1, 1, *list(w_tmp.shape[-(n-1):]))
            out = conv6d(x_tmp, w_tmp)
            index = torch.Tensor(range(w_size)).long().cuda()
            index = index.view(-1, 1)
            index = torch.cat([index]*np.prod(out.shape[2:]), -1)

            index = index.view([w_size, 1, *list(out.shape[2:])])
            out1 = out[:-1].gather(1, index)
            out2 = out[1:].gather(1, index)
            out1 = out1.sum(0)
            out2 = out2.sum(0)
            out = torch.cat([out1, out2], 0)
            out_b.append(out.view(1, *list(out.shape)))
        out_all.append(torch.cat(out_b, dim=0).view(1, -1, *list(out.shape)))
    out = torch.cat(out_all, dim=0)
    return out


if __name__ == '__main__':
    n = 1
    wW = 5
#    x = torch.Tensor(range(16)).cuda().view(1, 1, 4, 4)
#    w = torch.Tensor(range(9)).cuda().view(1, 1, 3, 3)
#    conv3d(x, w)
    for n in [1, 2, 3, 4, 5, 6, 7]:
        print('------------ ' + str(n) + ' -------------')
        torch.manual_seed(11)
        torch.cuda.manual_seed(11)
        x_size = [1, 1] + [wW + 1] * n
        w_size = [1, 1] + [wW] * n
        x = torch.rand(x_size).cuda()
        w = torch.rand(w_size).cuda()
        if n == 1:
            out0 = F.conv1d(x.double(), w.double())
            out1 = F.conv1d(x, w)
        elif n == 2:
            out0 = F.conv2d(x.double(), w.double())
            out1 = F.conv2d(x, w)
        elif n == 3:
            out0 = F.conv3d(x.double(), w.double())
            out1 = F.conv3d(x, w)
        elif n == 4:
            out0 = conv4d(x.double(), w.double())
            out1 = conv4d(x, w)
        elif n == 5:
            out0 = conv5d(x.double(), w.double())
            out1 = conv5d(x, w)
        elif n == 6:
            out0 = conv6d(x.double(), w.double())
            out1 = conv6d(x, w)
        elif n == 7:
            out0 = conv7d(x.double(), w.double())
            out1 = conv7d(x, w)
        out2 = matrix_convnd.conv32(x, w, n) 
        out3 = dwm_convnd.conv32(x, w, n)
        print(((out0 - out1.double()) / out0).abs().mean())
        print(((out0 - out2.double()) / out0).abs().mean())
        print(((out0 - out3.double()) / out0).abs().mean())
