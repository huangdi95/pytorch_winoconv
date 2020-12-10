import torch as th
from MakePytorchPlusPlus import wino_split, wino_concat
import os
import sys
from transform import Transform
from dwm import dwm

class Wino(Transform):
    def __init__(self):
        self.input= None
        self.kernel = None
        self.m = 0
        self.r = 0
        super(Wino, self).__init__()

    def matrix_alloc(m, r):
        A_ = []
        B_ = []
        G_ = []
        if (m[0], r[0]) in A:
            A_.append(tf.convert_to_tensor(A[(m[0], r[0])]))
            B_.append(tf.convert_to_tensor(B[(m[0], r[0])]))
            G_.append(tf.convert_to_tensor(G[(m[0], r[0])]))
        else:
            print('Size Error: F('+str(r[0])+', '+str(m[0])+') not implemented yet.')
            sys.exit(0)
        if (m[1], r[1]) in A:
            A_.append(tf.convert_to_tensor(A[(m[1], r[1])]))
            B_.append(tf.convert_to_tensor(B[(m[1], r[1])]))
            G_.append(tf.convert_to_tensor(G[(m[1], r[1])]))
        else:
            print('Size Error: F('+str(r[1])+', '+str(m[1])+') not implemented yet.')
            sys.exit(0)
        if (m[2], r[2]) in A:
            A_.append(tf.convert_to_tensor(A[(m[2], r[2])]))
            B_.append(tf.convert_to_tensor(B[(m[2], r[2])]))
            G_.append(tf.convert_to_tensor(G[(m[2], r[2])]))
        else:
            print('Size Error: F('+str(r[2])+', '+str(m[2])+') not implemented yet.')
            sys.exit(0)
        return A_, B_, G_

    def conv16(self, input, kernel, bias=None, stride=1, padding=0, groups=1, output_block=2):
        if stride != 1:
            if fp16:
                input = input.half()
                kernel = kernel.half()
            out = th.nn.functional.conv2d(input, kernel, bias=None, stride=stride, padding=0, dilation=1, groups=1)
            return out
        if type(padding) == type((1, 2)):
            pad = (padding[1], padding[1], padding[0], padding[0])
            input = th.nn.functional.pad(input, pad)
        #(N, H, W, C)
        input = input.cuda().float()
        kernel = kernel.cuda().float()
        x = input
        w = kernel
        out= self.conv(x, w, output_block, stride, True)
        if not bias is None:
            out = out.permute(0, 2, 3, 1)
            out = out.half() + bias.half() 
            out = out.permute(0, 3, 1, 2)
        out = out.float()
        return out

    def conv32(self, input, kernel, bias, stride=(1, 1), padding=0, groups=1, output_block=2):
        if stride[0] != 1:
            if fp16:
                input = input.half()
                kernel = kernel.half()
            out = th.nn.functional.conv2d(input, kernel, bias=None, stride=stride, padding=0, dilation=1, groups=1)
            return out
        if type(padding) == type((1, 2)):
            pad = (padding[1], padding[1], padding[0], padding[0])
            input = th.nn.functional.pad(input, pad)
        #(N, H, W, C)
        input = input.cuda().float()
        kernel = kernel.cuda().float()
        x = input
        w = kernel
        out= self.conv(x, w, output_block, stride, False)
        out = out.float()
        if not bias is None:
            out = out.permute(0, 2, 3, 1)
            out = out + bias 
            out = out.permute(0, 3, 1, 2)
        return out

    def conv(self, input, kernel, output_block, stride, fp16=False):
        #(K, C, H, W)
        w_shape = list(kernel.size())
        x_shape = list(input.size())
        H = x_shape[2]
        W = x_shape[3]
        wH = w_shape[2]
        wW = w_shape[3]
        
        stride = 1
        out_shape = [(H - wH) / stride + 1, (W - wW) / stride + 1]
        m = [output_block, output_block]
        r = [wH, wW]
        self.m = output_block
#            if fp16:
#                input = input.half()
#                kernel = kernel.half()
#            out = th.nn.functional.conv2d(input, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1)
#            return out
        #(N, C, H, W)
        input = wino_split(input, w_shape, output_block)
        if fp16:
            input = input.half()
            kernel = kernel.half()
        out = self.base_conv32(input, kernel, fp16)
        if fp16:
            out = out.float()
        #(N, H, W, C)
        concat = wino_concat(out, out_shape, output_block)
        return concat

    def base_conv32(self, input, kernel, fp16):
        #(N*nH*nW, C, 4, 4)
        A, B, G = self.matrix_alloc(input, kernel)
        #TODO is this reasonable?
        for i in range(3):
            A[i] = A[i].type_as(input)
            B[i] = B[i].type_as(input)
            G[i] = G[i].type_as(input)
#        kernel = kernel.permute(2, 3, 1, 0)

        A_shape = [list(A[0].size()), list(A[1].size()), list(A[2].size())]
        # [K, C, H, W]
        w_shape = list(kernel.size())
        x_shape = list(input.size())

        w_trans = G[0] @ kernel @ G[1].transpose(0, 1)
        
        w_trans = th.reshape(w_trans, (w_shape[0], w_shape[1], -1))
        w_trans = w_trans.permute(2, 0, 1)

        
        x_trans = B[0] @ input @ B[1].transpose(0, 1)

#        x_trans = th.reshape(input, (-1, x_shape[3]))
#        if(x_shape[3] != 2):
#            B_trans = th.transpose(B, 0, 1)
#            if fp16 == False:
#                x_trans = th.matmul(x_trans, B_trans)
#            else:
#                times = int(list(x_trans.size())[0]/8300000)
#                modu = int(list(x_trans.size())[0]%8300000)
#                if times > 0:
#                    tmp2 = th.matmul(x_trans[:8300000], B_trans)
#                    for i in list(range(times-1)):
#                        tmp1 = th.matmul(x_trans[((i+1)*8300000):((i+2)*8300000)], B_trans)
#                        tmp2 = th.cat((tmp2, tmp1), 0)
#                    tmp1 = th.matmul(x_trans[(times*8300000):(times*8300000+modu)], B_trans)
#                    x_trans = th.cat((tmp2, tmp1), 0)
#                else:
#                    x_trans = th.matmul(x_trans, B_trans)
        #print('2')
        #print(x_trans.shape)
#        x_trans = th.reshape(x_trans, (x_shape[0], x_shape[1], x_shape[2], -1))
#        #print('3')
#        #print(x_trans.shape)
#        x_trans = x_trans.permute(2, 3, 1, 0)
#        #print('4')
#        #print(x_trans.shape)
#        x_trans = th.reshape(x_trans, (x_shape[2], -1))
#        #print('5')
#        #print(x_trans.shape)
#        if(x_shape[2] != 2):
#            x_trans = th.matmul(B, x_trans)
#        #print('6')
#        #print(x_trans.shape)
#        x_trans = th.reshape(x_trans, (-1, x_shape[1], x_shape[0]))
        #print('7')
        #print(x_trans.shape)

        x_trans = th.reshape(x_trans, [x_shape[0], x_shape[1], -1])
        x_trans = x_trans.permute([2, 1, 0])

        mul = th.matmul(w_trans, x_trans)
        #print('8')
        #print(mul.shape)

        mul = th.reshape(mul, [A_shape[0][1], A_shape[1][1], w_shape[1], x_shape[0]])
        mul = mul.permute([4, 3, 0, 1])
        mul = A[0] @ mul @ A[1].transpose(0, 1)

#        if(x_shape[2] != 2):
#            mul = th.reshape(mul, (A_shape[1], -1))
#            mul = th.matmul(A, mul)
#        if(x_shape[3] != 2):
#            #(2, 4, K, N)
#            mul = th.reshape(mul, (A_shape[0], A_shape[1], w_shape[3], x_shape[0]))
#        else:
#            mul = th.reshape(mul, (A_shape[0], 2, w_shape[3], x_shape[0]))
#        #print('9')
#        #print(mul.shape)
#        #(N, K, 2, 4)
#        mul = mul.permute(3, 2, 0, 1)
#        mul = th.reshape(mul, (-1, list(mul.size())[-1]))
#        #print('10')
#        #print(mul.shape)
#        #(N * K * 2, 2)
#        if(x_shape[3] != 2):
#            A_trans = th.transpose(A, 0, 1)
#            if fp16 == False:
#                mul = th.matmul(mul, A_trans)
#            else:
#                times = int(list(mul.size())[0]/8000000)
#                modu = int(list(mul.size())[0]%8000000)
#                if times > 0:
#                    tmp2 = th.matmul(mul[:8000000], A_trans)
#                    for i in list(range(times-1)):
#                        #print(i)
#                        tmp1 = th.matmul(mul[((i+1)*8000000):((i+2)*8000000)], A_trans)
#                        tmp2 = th.cat((tmp2, tmp1), 0)
#                    tmp1 = th.matmul(mul[(times*8000000):(times*8000000+modu)], A_trans)
#                    mul = th.cat((tmp2, tmp1), 0)
#                else:
#                    mul = th.matmul(mul, A_trans)
        #print('11')
        #print(mul.shape)
        mul = th.reshape(mul, (x_shape[0], w_shape[3], A_shape[0][0], A_shape[1][0]))
        #print('12')
        #print(mul.shape)
        return mul 

