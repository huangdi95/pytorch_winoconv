import torch as th
import os
import sys
import numpy as np
from sympy import Rational
import wincnn
from MakePytorchPlusPlus import wino_split, wino_concat

input= None
kernel = None

A = {}
G = {}
B = {}
A[(2, 1)] = np.array([[1., 0.],
                [0., 1.]]).astype('float64')
G[(2, 1)] = np.array([[  1.],
                [  1.]]).astype('float64')
B[(2, 1)] = np.array([[1., 0],
                [0., 1.]]).astype('float64')
A[(2, 3)] = np.array([[1., 1., 1., 0.],
                [0., 1.,-1.,-1.]]).astype('float64')
G[(2, 3)] = np.array([[  1.,   0.,   0.],
                [1/2., 1/2., 1/2.],
                [1/2.,-1/2., 1/2.],
                [  0.,   0.,   1.]]).astype('float64')
B[(2, 3)] = np.array([[1., 0.,-1., 0.],
                [0., 1., 1., 0.],
                [0.,-1., 1., 0.],
                [0., 1., 0.,-1.]]).astype('float64')
points = (0,1,-1,Rational(1,2),-Rational(1,2),Rational(1,3),-Rational(1,3),Rational(3,2),-Rational(3,2),-3,2,-2)
A[(2, 4)], G[(2, 4)], B[(2, 4)] = wincnn.showCookToomFilter((0,1,-1,2), 2, 4)
A[(2, 5)], G[(2, 5)], B[(2, 5)] = wincnn.showCookToomFilter((0,1,-1,2,-2), 2, 5)
A[(2, 6)], G[(2, 6)], B[(2, 6)] = wincnn.showCookToomFilter(points, 2, 6)
A[(2, 7)], G[(2, 7)], B[(2, 7)] = wincnn.showCookToomFilter(points, 2, 7)
A[(2, 8)], G[(2, 8)], B[(2, 8)] = wincnn.showCookToomFilter(points, 2, 8)
A[(2, 9)], G[(2, 9)], B[(2, 9)] = wincnn.showCookToomFilter(points, 2, 9)
A[(2,10)], G[(2,10)], B[(2,10)] = wincnn.showCookToomFilter(points, 2,10)
A[(2,11)], G[(2,11)], B[(2,11)] = wincnn.showCookToomFilter(points, 2,11)
A[(9, 5)], G[(9, 5)], B[(9, 5)] = wincnn.showCookToomFilter(points, 9, 5)

def matrix_alloc(m, r):
    A_ = []
    B_ = []
    G_ = []
    if (m[0], r[0]) in A:
        A_.append(th.from_numpy(A[(m[0], r[0])]).cuda())
        B_.append(th.from_numpy(B[(m[0], r[0])]).cuda())
        G_.append(th.from_numpy(G[(m[0], r[0])]).cuda())
    else:
        print('Size Error: F('+str(r[0])+', '+str(m[0])+') not implemented yet.')
        sys.exit(0)
    if (m[1], r[1]) in A:
        A_.append(th.from_numpy(A[(m[1], r[1])]).cuda())
        B_.append(th.from_numpy(B[(m[1], r[1])]).cuda())
        G_.append(th.from_numpy(G[(m[1], r[1])]).cuda())
    else:
        print('Size Error: F('+str(r[1])+', '+str(m[1])+') not implemented yet.')
        sys.exit(0)
    if len(m) > 2:
        if (m[2], r[2]) in A:
            A_.append(th.from_numpy(A[(m[2], r[2])]).cuda())
            B_.append(th.from_numpy(B[(m[2], r[2])]).cuda())
            G_.append(th.from_numpy(G[(m[2], r[2])]).cuda())
        else:
            print('Size Error: F('+str(r[2])+', '+str(m[2])+') not implemented yet.')
            sys.exit(0)
    return A_, B_, G_

def conv16(input, kernel, bias=None, stride=(1, 1), padding=0, groups=1, output_block=2):
    if sum(stride) != len(stride):
        input = input.half()
        kernel = kernel.half()
        out = th.nn.functional.conv2d(input, kernel, bias=bias, stride=stride, padding=padding, dilation=1, groups=1)
        return out
    if type(padding) == type((1, 2)):
        pad = (padding[1], padding[1], padding[0], padding[0])
        input = th.nn.functional.pad(input, pad)
    #(N, H, W, C)
    input = input.cuda().float()
    kernel = kernel.cuda().float()
    x = input
    w = kernel
    out= conv(x, w, output_block, stride, True)
    if not bias is None:
        out = out.permute(0, 2, 3, 1)
        out = out.half() + bias.half() 
        out = out.permute(0, 3, 1, 2)
    out = out.float()
    return out

def conv32(input, kernel, bias=None, stride=(1, 1), padding=0, groups=1, output_block=2):
    if sum(stride) != len(stride):
        out = th.nn.functional.conv2d(input, kernel, bias=bias, stride=stride, padding=padding, dilation=1, groups=1)
        return out
    if type(padding) == type((1, 2)):
        pad = (padding[1], padding[1], padding[0], padding[0])
        input = th.nn.functional.pad(input, pad)
    #(N, H, W, C)
    input = input.cuda().float()
    kernel = kernel.cuda().float()
    x = input
    w = kernel
    out= conv(x, w, output_block, stride, False)
    out = out.float()
    if not bias is None:
        out = out.permute(0, 2, 3, 1)
        out = out + bias 
        out = out.permute(0, 3, 1, 2)
    return out

def conv(input, kernel, output_block, stride, fp16=False):
    #(K, C, H, W)
    w_shape = list(kernel.size())
    x_shape = list(input.size())
    H = x_shape[2]
    W = x_shape[3]
    wH = w_shape[2]
    wW = w_shape[3]
    
    stride = 1
    out_shape = [int((H - wH) / stride + 1), int((W - wW) / stride + 1)]
    m = [output_block, output_block]
    r = [wH, wW]
    #(N, C, H, W)
    input = wino_split(input, w_shape, output_block)
    if fp16:
        input = input.half()
        kernel = kernel.half()
    out = base_conv32(input, kernel, fp16, m, r)
    if fp16:
        out = out.float()
    #(N, H, W, C)
    concat = wino_concat(out, out_shape, output_block)
    return concat

def base_conv32(input, kernel, fp16, m, r):
    #(N*nH*nW, C, 4, 4)
    A, B, G = matrix_alloc(m, r)
    for i in range(2):
        A[i] = A[i].type_as(input)
        B[i] = B[i].type_as(input)
        G[i] = G[i].type_as(input)

    A_shape = [list(A[0].size()), list(A[1].size())]
    # [K, C, H, W]
    w_shape = list(kernel.size())
    x_shape = list(input.size())

    w_trans = G[0] @ kernel @ G[1].transpose(0, 1)
    
    w_trans = th.reshape(w_trans, (w_shape[0], w_shape[1], -1))
    w_trans = w_trans.permute(2, 0, 1)

    
    max_size = 8000000
    if not fp16:
        x_trans = B[0] @ input @ B[1].transpose(0, 1)
    else:
        B_trans = th.transpose(B[1], 0, 1)
        x_trans = th.reshape(input, (-1, x_shape[3]))
        times = int(list(x_trans.size())[0]/max_size)
        modu = int(list(x_trans.size())[0]%max_size)
        if times > 0:
            tmp2 = th.matmul(x_trans[:max_size], B_trans)
#            tmp2 = B[0] @ x_trans[:max_size] @ B[1].transpose(0, 1)
            for i in list(range(times-1)):
                tmp1 = th.matmul(x_trans[((i+1)*max_size):((i+2)*max_size)], B_trans)
#                tmp1 = B[0] @ x_trans[((i+1)*max_size):((i+2)*max_size)] @ B[1].transpose(0, 1)
                tmp2 = th.cat((tmp2, tmp1), 0)
            tmp1 = th.matmul(x_trans[(times*max_size):(times*max_size+modu)], B_trans)
#            tmp1 = B[0] @ x_trans[(times*max_size):(times*max_size+modu)] @ B[1].transpose(0, 1)
            x_trans = th.cat((tmp2, tmp1), 0)
        else:
#            x_trans = B[0] @ input @ B[1].transpose(0, 1)
            x_trans = th.matmul(x_trans, B_trans)
        x_trans = x_trans.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])

        B_trans = th.transpose(B[0], 0, 1)
        x_trans = th.transpose(x_trans, -2, -1)
        x_trans = th.reshape(x_trans, (-1, x_shape[2]))
        times = int(list(x_trans.size())[0]/max_size)
        modu = int(list(x_trans.size())[0]%max_size)
        if times > 0:
            tmp2 = th.matmul(x_trans[:max_size], B_trans)
#            tmp2 = B[0] @ x_trans[:max_size] @ B[1].transpose(0, 1)
            for i in list(range(times-1)):
                tmp1 = th.matmul(x_trans[((i+1)*max_size):((i+2)*max_size)], B_trans)
#                tmp1 = B[0] @ x_trans[((i+1)*max_size):((i+2)*max_size)] @ B[1].transpose(0, 1)
                tmp2 = th.cat((tmp2, tmp1), 0)
            tmp1 = th.matmul(x_trans[(times*max_size):(times*max_size+modu)], B_trans)
#            tmp1 = B[0] @ x_trans[(times*max_size):(times*max_size+modu)] @ B[1].transpose(0, 1)
            x_trans = th.cat((tmp2, tmp1), 0)
        else:
#            x_trans = B[0] @ input @ B[1].transpose(0, 1)
            x_trans = th.matmul(x_trans, B_trans)
        x_trans = x_trans.view(x_shape[0], x_shape[1], x_shape[3], x_shape[2])
        x_trans = th.transpose(x_trans, -2, -1)
#        x_trans = B[0] @ x_trans 


    x_trans = th.reshape(x_trans, [x_shape[0], x_shape[1], -1])
    x_trans = x_trans.permute([2, 1, 0])

    mul = th.matmul(w_trans, x_trans)

    mul = th.reshape(mul, [A_shape[0][1], A_shape[1][1], w_shape[0], x_shape[0]])
    mul = mul.permute([3, 2, 0, 1])

    if not fp16:
        mul = A[0] @ mul @ A[1].transpose(0, 1)
    else:
        A_trans = th.transpose(A[1], 0, 1)
        mul = th.reshape(mul, (-1, A_shape[1][1]))
        times = int(list(mul.size())[0]/max_size)
        modu = int(list(mul.size())[0]%max_size)
        if times > 0:
            tmp2 = th.matmul(mul[:max_size], A_trans)
#            tmp2 = A[0] @ mul[:max_size] @ A[1].transpose(0, 1)
            for i in list(range(times-1)):
                tmp1 = th.matmul(mul[((i+1)*max_size):((i+2)*max_size)], A_trans)
#                tmp1 = A[0] @ mul[((i+1)*max_size):((i+2)*max_size)] @ A[1].transpose(0, 1)
                tmp2 = th.cat((tmp2, tmp1), 0)
            tmp1 = th.matmul(mul[(times*max_size):(times*max_size+modu)], A_trans)
#            tmp1 = A[0] @ mul[(times*max_size):(times*max_size+modu)] @ A[1].transpose(0, 1)
            mul = th.cat((tmp2, tmp1), 0)
        else:
#            mul = A[0] @ mul @ A[1].transpose(0, 1)
            mul = th.matmul(mul, A_trans)
        mul = mul.view(x_shape[0], w_shape[0], A_shape[0][1], A_shape[1][0])

        A_trans = th.transpose(A[0], 0, 1)
        mul = th.transpose(mul, -2, -1)
        mul = th.reshape(mul, (-1, A_shape[0][1]))
        times = int(list(mul.size())[0]/max_size)
        modu = int(list(mul.size())[0]%max_size)
        if times > 0:
            tmp2 = th.matmul(mul[:max_size], A_trans)
#            tmp2 = A[0] @ mul[:max_size] @ A[1].transpose(0, 1)
            for i in list(range(times-1)):
                tmp1 = th.matmul(mul[((i+1)*max_size):((i+2)*max_size)], A_trans)
#                tmp1 = A[0] @ mul[((i+1)*max_size):((i+2)*max_size)] @ A[1].transpose(0, 1)
                tmp2 = th.cat((tmp2, tmp1), 0)
            tmp1 = th.matmul(mul[(times*max_size):(times*max_size+modu)], A_trans)
#            tmp1 = A[0] @ mul[(times*max_size):(times*max_size+modu)] @ A[1].transpose(0, 1)
            mul = th.cat((tmp2, tmp1), 0)
        else:
#            mul = A[0] @ mul @ A[1].transpose(0, 1)
            mul = th.matmul(mul, A_trans)
        mul = mul.view(x_shape[0], w_shape[0], A_shape[1][0], A_shape[0][0])
        mul = th.transpose(mul, -2, -1)

    mul = th.reshape(mul, (x_shape[0], w_shape[0], A_shape[0][0], A_shape[1][0]))
    return mul
