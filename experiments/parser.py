#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Sat 05 Dec 2020 11:02:43 PM CST
#########################################################################
import os
import csv

def get_time(line):
    perc = 0.
    time = 0.
    for item in line.split():
        if '%' in item:
            perc = float(item[:-1])
        elif 'us' in item:
            time = float(item[:-2])/1000
            break
        elif 'ms' in item:
            time = float(item[:-2])
            break
        elif 's' in item and 'activit' not in item:
            time = float(item[:-2])*1000
            break
    return perc, time


f = open('time.csv','w+')
writer = csv.writer(f)
writer.writerow(['name', 'S', 'N', 'C', 'K', 'D', 'H', 'W', 'wD', 'wH', 'wW', 'cudnn', 'memset', 'memcpy', '', 'xtrans', 'wtrans', 'otrans', 'oadd', 'gemm', 'memset', 'memcpy', '', 'split', 'concat', 'meta', 'shuffle', 'swap', 'gemm', 'memset', 'memcpy'])
for root,dirs,files in os.walk(r"./cudnn/logs"):
    for file in files:
        name = os.path.join(root,file)
        if 'sw' in name : continue
        print(name) 
        head = name.split('.')[-1]
        row = [head]
        row.extend(head.split('_'))
        with open(name, 'r') as f:
            cudnn = 0.
            memcpy = 0.
            memset = 0.
            for l in f.readlines():
                if 'Error' in l or 'ERROR' in l or 'error' in l:
                    continue 
                if 'cudnn' in l:
                    cudnn += get_time(l)[1]
                elif 'memset' in l:
                    memset += get_time(l)[1]
                elif 'memcpy' in l:
                    memcpy += get_time(l)[1]
                if 'API calls' in l:
                    break
            row.extend([cudnn, memset, memcpy])
#            print(cudnn)
#            print(memcpy)
#            print(memset)
        dwmname = './dwm/logs/log.' + head
#        print(dwmname)
        if not os.path.exists(dwmname):
            continue
        with open(dwmname, 'r') as f:
            xtrans = 0.
            wtrans = 0.
            otrans = 0.
            oadd = 0.
            gemm = 0.
            memset = 0.
            memcpy = 0.
            for l in f.readlines():
                if 'Error' in l or 'ERROR' in l or 'error' in l:
                    continue 
                if 'inputNorm2Wino' in l:
                    xtrans += get_time(l)[1]
                elif 'wNorm2Wino' in l:
                    wtrans += get_time(l)[1]
                elif 'outputWino2Norm' in l:
                    otrans += get_time(l)[1]
                elif 'outputAggregate' in l:
                    oadd += get_time(l)[1]
                elif 'gemm' in l:
                    gemm += get_time(l)[1]
                elif 'memcpy' in l:
                    memcpy += get_time(l)[1]
                elif 'memset' in l:
                    memset += get_time(l)[1]
                if 'API calls' in l:
                    break
            
            row.extend(['', xtrans, wtrans, otrans, oadd, gemm, memset, memcpy])
#            print(wtrans)
#            print(otrans)
#            print(oadd)
#            print(gemm)
#            print(memcpy)
#            print(memset)

        winoname = './winograd/logs/log.' + head
#        print(winoname)
        if not os.path.exists(winoname):
            continue
        with open(winoname, 'r') as f:
            split = 0.
            concat = 0.
            meta = 0.
            shuffle = 0.
            swap = 0.
            gemm = 0.
            memset = 0.
            memcpy = 0.
            for l in f.readlines():
                if 'Error' in l or 'ERROR' in l or 'error' in l:
                    continue 
                if 'winoSplitKernel' in l:
                    split += get_time(l)[1]
                elif 'winoConcatKernel' in l:
                    concat += get_time(l)[1]
                elif 'EigenMetaKernel' in l:
                    meta += get_time(l)[1]
                elif 'ShuffleInTensor' in l:
                    shuffle += get_time(l)[1]
                elif 'SwapDimension' in l:
                    swap += get_time(l)[1]
                elif 'gemm' in l:
                    gemm += get_time(l)[1]
                elif 'memcpy' in l:
                    memcpy += get_time(l)[1]
                elif 'memset' in l:
                    memset += get_time(l)[1]
                if 'API calls' in l:
                    break
            
            row.extend(['', split, concat, meta, shuffle, swap, gemm, memset, memcpy])
        if 0 in row:
            continue
        writer.writerow(row)
