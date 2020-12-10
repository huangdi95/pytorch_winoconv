// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cublas_v2.h>

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)
// I = (Batch, H, W, C)
// O = (Batch*nH*nW, C, 4, 4)
template <typename T>
__global__ void winoSplitKernel(const T *norm_input, int B, int D, int H, int W, int C, int pad_d, int pad_h, int pad_w, int block_D, int block_H, int block_W, int nD, int nH, int nW, T *wino_input)
{ 
    int bx = blockIdx.x; // nd*nh*nw
    int by = blockIdx.y; // b
    int t = threadIdx.x; // c

    int d = bx / (nH * nW); 
    int h = (bx % (nH * nW)) / nW; 
    int w = (bx % (nH * nW)) % nW;
    
    int f_b = by;
    int xBase = 2 * w - pad_w;
    int yBase = 2 * h - pad_h;
    int zBase = 2 * d - pad_d;
    
    T input_patch;

    int f_x, f_y, f_z;
    for(int k = 0; k < block_D; k++) {
        for(int i = 0; i < block_H; i++) {
            for(int j = 0; j < block_W; j++) {
                f_x = xBase + j; f_y = yBase + i; f_z = zBase + k;
                if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H) && (f_z > -1) && (f_z < D)) input_patch = norm_input [((((f_b * D + f_z) * H) + f_y) * W + f_x) * C + t]; 
                else input_patch = T(0);
                wino_input[((((((by * nD + d) * nH + h) * nW + w) * C + t) * block_D + k) * block_H + i) * block_W + j] = T(input_patch);
            }
        }
     } 
} 

// dim3 threadsPerBlock(K)
// dim3 numBlocks(Batch, nH, nW)
// wino_output = (Batch*nH*nW, K, 4)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void winoConcatKernel(const T *wino_output, int B, int output_D, int output_H, int output_W, int K, int output_block_D, int output_block_H, int output_block_W, T *norm_output)
{
    int bx = blockIdx.x; // nd*nh*nw
    int by = blockIdx.y; // b
    int tx = threadIdx.x; // c
    
    int nD;
    int nH;
    int nW;
    nD = (output_D+output_block_D-1)/output_block_D;
    nH = (output_H+output_block_H-1)/output_block_H;
    nW = (output_W+output_block_W-1)/output_block_W;

    int d = bx / (nH * nW); 
    int h = (bx % (nH * nW)) / nW; 
    int w = (bx % (nH * nW)) % nW;

    for(int k = 0; k < output_block_D; k++) {
        for(int i = 0; i < output_block_H; i++) {
            for(int j = 0; j < output_block_W; j++) {
                T output_patch = wino_output[((((((by * nD + d) * nH + h) * nW + w) * K + tx) * output_block_D + k) * output_block_H + i) * output_block_W + j];
                if((k <= (output_D - 1) % output_block_D || k != nD - 1) && (i <= (output_H - 1) % output_block_H || h != nH - 1) && (j <= (output_W - 1) % output_block_W || w != nW - 1))
                    norm_output[by*output_D*output_H*output_W*K + (output_block_D*d+k)*output_H*output_W*K + (output_block_H*h+i)*output_W*K + (output_block_W*w+j)*K + tx] = output_patch;
            }
        }
    }
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)
// I = (Batch, H, W, C)
// O = (Batch*nH*nW, C, 4, 4)
template <typename T>
__global__ void winoSplitKernel2d(const T *norm_input, int B, int H, int W, int C, int pad_h, int pad_w, int block_H, int block_W, int nH, int nW, T *wino_input)
{ 
    int bx = blockIdx.x; // nh*nw
    int by = blockIdx.y; // b
    int t = threadIdx.x; // c

    int h = bx / nW; 
    int w = bx % nW;
    
    int f_b = by;
    int xBase = 2 * w - pad_w;
    int yBase = 2 * h - pad_h;
    
    T input_patch;

    int f_x, f_y;
    for(int i = 0; i < block_H; i++) {
        for(int j = 0; j < block_W; j++) {
            f_x = xBase + j; f_y = yBase + i;
            if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch = norm_input [(((f_b * H) + f_y) * W + f_x) * C + t]; 
            else input_patch = T(0);
            wino_input[((((by * nH + h) * nW + w) * C + t) * block_H + i) * block_W + j] = T(input_patch);
        }
    }
} 

// dim3 threadsPerBlock(K)
// dim3 numBlocks(Batch, nH, nW)
// wino_output = (Batch*nH*nW, K, 4)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void winoConcatKernel2d(const T *wino_output, int B, int output_H, int output_W, int K, int output_block_H, int output_block_W, T *norm_output)
{
    int bx = blockIdx.x; // nh*nw
    int by = blockIdx.y; // b
    int tx = threadIdx.x; // c
    
    int nH;
    int nW;
    nH = (output_H+output_block_H-1)/output_block_H;
    nW = (output_W+output_block_W-1)/output_block_W;

    int h = bx / nW; 
    int w = bx % nW;

    for(int i = 0; i < output_block_H; i++) {
        for(int j = 0; j < output_block_W; j++) {
            T output_patch = wino_output[((((by * nH + h) * nW + w) * K + tx) * output_block_H + i) * output_block_W + j];
            if((i <= (output_H - 1) % output_block_H || h != nH - 1) && (j <= (output_W - 1) % output_block_W || w != nW - 1))
                norm_output[by*output_H*output_W*K + (output_block_H*h+i)*output_W*K + (output_block_W*w+j)*K + tx] = output_patch;
        }
    }
}
