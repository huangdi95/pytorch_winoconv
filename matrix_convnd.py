import torch as th
import os
import sys
import numpy as np
from sympy import Rational
import wincnn

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
A[(2, 2)] = np.array([[1., 1., 0.],
                [0., 1., 1.]]).astype('float64')
G[(2, 2)] = np.array([[  1.,   0.],
                [1., 1.],
                [0.,   1.]]).astype('float64')
B[(2, 2)] = np.array([[1., -1., 0.],
                [0., 1., 0.],
                [0., -1., 1.]]).astype('float64')
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

    for i_r in r:
        A_.append(th.from_numpy(A[(2, i_r)]))
        B_.append(th.from_numpy(B[(2, i_r)]))
        G_.append(th.from_numpy(G[(2, i_r)]))
#    if (m, r) in A:
#        A_=(th.from_numpy(A[(m, r)]).cuda())
#        B_=(th.from_numpy(B[(m, r)]).cuda())
#        G_=(th.from_numpy(G[(m, r)]).cuda())
#    else:
#        print('Size Error: F('+str(r[0])+', '+str(m[0])+') not implemented yet.')
#        sys.exit(0)
    return A_, B_, G_

#def conv16(input, kernel, bias=None, stride=(1, 1), padding=0, groups=1, output_block=2):
#    if sum(stride) != len(stride):
#        input = input.half()
#        kernel = kernel.half()
#        out = th.nn.functional.conv2d(input, kernel, bias=bias, stride=stride, padding=padding, dilation=1, groups=1)
#        return out
#    if type(padding) == type((1, 2)):
#        pad = (padding[1], padding[1], padding[0], padding[0])
#        input = th.nn.functional.pad(input, pad)
#    #(N, H, W, C)
#    input = input.cuda().float()
#    kernel = kernel.cuda().float()
#    x = input
#    w = kernel
#    out= conv(x, w, output_block, stride, True)
#    if not bias is None:
#        out = out.permute(0, 2, 3, 1)
#        out = out.half() + bias.half() 
#        out = out.permute(0, 3, 1, 2)
#    out = out.float()
#    return out

def conv32(input, kernel, n, output_block=2):
    #(N, H, W, C)
    input = input.cuda().float()
    kernel = kernel.cuda().float()
    x = input
    w = kernel
    out= conv(x, w, n, output_block, False)
    out = out.float()
    return out

def conv(input, kernel, n, output_block=2, fp16=False):
    #(K, C, H, W)
    w_shape = list(kernel.size())
    
    m = 2
    r = w_shape[2:]
    #(N, C, H, W)
    out = base_conv32(input, kernel, fp16, m, r, n)
    #(N, H, W, C)
    return out 

def base_conv32(input, kernel, fp16, m, r, n):
    #(N*nH*nW, C, 4, 4)
    A, B, G = matrix_alloc(m, r)
    for i in range(len(A)):
        A[i] = A[i].type_as(input)
        B[i] = B[i].type_as(input)
        G[i] = G[i].type_as(input)

    # [K, C, H, W]
    w_shape = list(kernel.size())
    x_shape = list(input.size())

    w_trans = kernel
    for i in range(n):
        w_trans = w_trans.transpose(-1, -(n-i))
        w_trans = w_trans @ G[i].transpose(0, 1)
        w_trans = w_trans.transpose(-1, -(n-i))
    
    w_trans = th.reshape(w_trans, (w_shape[0], w_shape[1], -1))
    w_trans = w_trans.permute(2, 0, 1)

    x_trans = input
    for i in range(n):
        x_trans = x_trans.transpose(-1, -(n-i))
        x_trans = x_trans @ B[i].transpose(0, 1)
        x_trans = x_trans.transpose(-1, -(n-i))

    x_trans = th.reshape(x_trans, [x_shape[0], x_shape[1], -1])
    x_trans = x_trans.permute([2, 1, 0])

    mul = th.matmul(w_trans, x_trans)


    A_shape = [i + 1 for i in r]
    mul = th.reshape(mul, A_shape + [w_shape[0], x_shape[0]])
    mul = mul.permute([n + 1, n] + list(range(n)))

    for i in range(n):
        mul = mul.transpose(-1, -(n-i))
        mul = mul @ A[i].transpose(0, 1)
        mul = mul.transpose(-1, -(n-i))

    mul = th.reshape(mul, [x_shape[0], w_shape[0]] + [2] * n)
    return mul
