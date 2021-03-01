import torch
import os
import sys
import numpy as np
import matrix_convnd

def conv32(input, kernel, n):
    w_shape = list(kernel.size())
    x_shape = list(input.size())
    times = (np.array(w_shape[2:]) / 3).astype(np.int)
    modu = (np.array(w_shape[2:]) % 3).astype(np.int)

    num = times[0] + int(modu[0]!=0)
    num_split = num**n
    idx_helper = []
    tmp = 1
    for i in range(n):
        tmp *= num
        idx_helper.append(tmp)
    idx_helper = np.array(idx_helper[::-1]).astype(np.int)


    out = torch.zeros([x_shape[0], w_shape[1]] + [2] * n).cuda()
    for i in range(num_split):
        location = (((i % idx_helper) / (idx_helper / num)).astype(np.int))

        w_idx = [slice(None, None), slice(None, None)]
        x_idx = [slice(None, None), slice(None, None)]
        for j in range(len(location)):
            if location[j] == num - 1 and modu[0] != 0: 
                w_idx.append(slice(location[j] * 3, location[j] * 3 + modu[0]))
                x_idx.append(slice(location[j] * 3, location[j] * 3 + modu[0]+1))
            else:
                w_idx.append(slice(location[j] * 3, location[j] * 3 + 3))
                x_idx.append(slice(location[j] * 3, location[j] * 3 + 4))

        w_tmp = kernel[w_idx]
        x_tmp = input[x_idx]
        out = out + matrix_convnd.conv32(x_tmp, w_tmp, n)
    return out
    
