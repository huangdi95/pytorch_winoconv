import torch
import MakePytorchPlusPlus as MPP
#from WinoConv import Wino
import dwm
import matrix_conv
import matrix_conv3d
#import matrix_conv3d

#
#def test_foo():
#    foo = MPP.Foo()
#    print(foo)
#    foo.setKey(3)
#    print(foo)
#    print(foo.getKey())
#
#
#def test_add_gpu():
#    if not torch.cuda.is_available():
#        return
#    a = torch.ones(4).cuda().half()
#    b = torch.ones(4).cuda().half()
#    c = MPP.add_gpu(a, b)
#    print(a, b, c)
#    print(c.shape)
#    print(c)
#
#
#def test_split():
#    if not torch.cuda.is_available():
#        return
#    a = torch.cuda.FloatTensor(1, 1, 6, 6)
#    a.normal_()
#    print(a)
#    c = MPP.wino_split(a, 4, 2)
#    print(c)
#    print(c.shape)
#
#def test_concat():
#    if not torch.cuda.is_available():
#        return
#    a = torch.cuda.FloatTensor(4, 2, 2, 2)
#    a.normal_()
#    print(a)
#    c = MPP.wino_concat(a, 3, 2)
#    print(c)
#    print(c.shape)
#
#def test_conv():
#    if not torch.cuda.is_available():
#        return
#    x = torch.tensor([
#                     [[[1, 2, 3, 4],
#                      [1, 2, 3, 4],
#                      [1, 2, 3, 4],
#                      [1, 2, 3, 4]],
#                     [[10, 20, 30, 40],
#                      [10, 200, 30, 40],
#                      [10, 20, 30, 40],
#                      [10, 20, 30, 40]]],
#                     [[[1, 2, 3, 4],
#                      [1, 2, 3, 4],
#                      [1, 2, 3, 4],
#                      [1, 2, 3, 4]],
#                     [[10, 20, 30, 40],
#                      [10, 20, 30, 40],
#                      [10, 20, 30, 40],
#                      [10, 20, 30, 40]]]
#                      ]).cuda().float()
#    g = 2
##    x = torch.rand(2, g, 4, 4).cuda().float()
#    w = torch.rand(g, 1, 3, 3).cuda().float()
#    out2 = torch.nn.functional.conv2d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=g)
#    wino = Wino()
#    out1 = wino.conv16(x, w, groups=g)
##    out2 = wino.conv32(x, w)
#    print(out1)
#    print(out2)
#    print(((out1 - out2)/out2).abs().max())
#    
#def test_matmul():
#    x_trans = torch.rand(24780800, 4).cuda().half()
#    B_trans = torch.rand(4, 4).cuda().half()
#    times = int(list(x_trans.size())[0]/8300000)
#    modu = int(list(x_trans.size())[0]%8300000)
#    if times > 1:
#        tmp2 = torch.matmul(x_trans[:8300000], B_trans)
#        for i in list(range(times-1)):
#            print(i)
#            tmp1 = torch.matmul(x_trans[((i+1)*8300000):((i+2)*8300000)], B_trans)
#            tmp2 = torch.cat((tmp2, tmp1), 0)
#        tmp1 = torch.matmul(x_trans[(times*8300000):(times*8300000+modu)], B_trans)
#        x_trans = torch.cat((tmp2, tmp1), 0)
#    else:
#        x_trans = torch.matmul(x_trans, B_trans)
#    print(x_trans)
#    print(x_trans.shape)

def test_dwm2d():
    x = torch.rand(1, 32*7, 10, 10).cuda()
    w = torch.rand(32, 32*7, 7, 7).cuda()
    out1 = dwm.dwm2d(x.half(), w.half(), None, (1, 1))
    out3 = matrix_conv.conv16(x.half(), w.half())
    out2 = torch.nn.functional.conv2d(x.double(), w.double(), bias=None, stride=1, padding=0, dilation=1, groups=1)
    print(((out1.double() - out2)/out2).abs().mean())
    print(((out3.double() - out2)/out2).abs().mean())
    print((out3.double()).abs().mean())
    print(((out1.double() - out3.double())/out1.double()).abs().mean())
    return out1.double()

def test_dwm3d():
    x = torch.rand(1, 32, 10, 10, 10).cuda()
    w = torch.rand(32, 32, 7, 7, 7).cuda()
    out1 = dwm.dwm3d(x.half(), w.half(), None, (1, 1, 1))
    out2 = torch.nn.functional.conv3d(x.double(), w.double(), bias=None, stride=1, padding=0, dilation=1, groups=1)
    print(((out1.double() - out2)/out2).abs().mean())
    return out1.double()

def test_matrix_conv2d():
    x = torch.rand(1, 32*7, 10, 10).cuda()
    w = torch.rand(32, 32*7, 7, 7).cuda()
 #   out0 = dwm.dwm2d(x, w, None, (1, 1))
    out1 = matrix_conv.conv32(x, w)
    out2 = torch.nn.functional.conv2d(x.double(), w.double(), bias=None, stride=1, padding=0, dilation=1, groups=1)
#    out3 = torch.nn.functional.conv2d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)
    print(((out1.double() - out2)/out2).abs().mean())
#    print(((out0.double() - out2)/out2).abs().mean())
#    print(((out3.double() - out2)/out2).abs().mean())

def test_matrix_conv3d():
    x = torch.rand(1, 32, 10, 10, 10).cuda()
    w = torch.rand(32, 32, 7, 7, 7).cuda()
    out1 = matrix_conv3d.conv32(x, w)
    out2 = torch.nn.functional.conv3d(x.double(), w.double(), bias=None, stride=1, padding=0, dilation=1, groups=1)
    print(((out1.double() - out2)/out2).abs().mean())

def test_matrix_conv2d_fp16():
    x = torch.rand(64, 32*7, 10, 10).cuda()
    w = torch.rand(32, 32*7, 7, 7).cuda()
    out1 = matrix_conv.conv16(x.half(), w.half())
    out2 = torch.nn.functional.conv2d(x.double(), w.double(), bias=None, stride=1, padding=0, dilation=1, groups=1)
    print(((out1.double() - out2)/out2).abs().mean())
    return out1.double()

def test_matrix_conv3d_fp16():
    x = torch.rand(1, 32, 10, 10, 10).cuda()
    w = torch.rand(32, 32, 7, 7, 7).cuda()
    out1 = matrix_conv3d.conv16(x.half(), w.half())
    out2 = torch.nn.functional.conv3d(x.double(), w.double(), bias=None, stride=1, padding=0, dilation=1, groups=1)
    print(((out1.double() - out2)/out2).abs().mean())
    

if __name__ == '__main__':
#    test_dwm2d()
#    test_dwm3d()
#    test_matrix_conv2d()
    test_matrix_conv3d()
 #   test_matrix_conv2d_fp16()
    test_matrix_conv3d_fp16()

