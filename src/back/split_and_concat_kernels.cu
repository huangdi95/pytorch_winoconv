#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)
// I = (Batch, H, W, C)
// O = (Batch*nH*nW, C, 4, 4)
template <typename T>
__global__ void winoSplitKernel(const T *norm_input, int B, int H, int W, int C, int pad_h, int pad_w, int block_H, int block_W, int nH, int nW, T *wino_input)
{ 
    int bx = blockIdx.x; // nw
    int by = blockIdx.y; // nh
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    float input_patch;

    int f_x, f_y;
    for(int i = 0; i < block_H; i++) {
        for(int j = 0; j < block_W; j++) {
            f_x = xBase + j; f_y = yBase + i;
            if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch = float(norm_input [ f_b * H * W * C + t * H * W + f_y * W + f_x]);
            else input_patch = 0;
            wino_input[((((bz * nH + by) * nW + bx) * C + t) * block_H + i) * block_W + j] = T(input_patch);
        }
    }
    
} 

// dim3 threadsPerBlock(K)
// dim3 numBlocks(Batch, nH, nW)
// wino_output = (Batch*nH*nW, K, 4)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void winoConcatKernel(const T *wino_output, int B, int output_H, int output_W, int K, int output_block_H, int output_block_W, T *norm_output)
{
    int bx = blockIdx.x; // nw
    int by = blockIdx.y; // nh
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+output_block_H-1)/output_block_H;
    nW = (output_W+output_block_W-1)/output_block_W;

    for(int i = 0; i < output_block_H; i++) {
        for(int j = 0; j < output_block_W; j++) {
            float output_patch = float(wino_output[((((bz * nH + by) * nW + bx) * K + tx) * output_block_H + i) * output_block_W + j]);
            if((i <= (output_H - 1) % output_block_H || by != nH - 1) && (j <= (output_W - 1) % output_block_W || bx != nW - 1))
                norm_output[bz*output_H*output_W*K + tx*output_H*output_W + (output_block_H*by+i)*output_W + (output_block_W*bx+j)] = T(output_patch);
        }
    }
}

