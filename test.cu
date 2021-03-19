/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Mon 15 Mar 2021 11:06:07 AM CST
 ************************************************************************/
#include <stdio.h>
#include <iostream>
#include "src/calculation_kernels2d.cu"

__global__ void read(float *wino_output, float *tmp_output, int *time, int B, int output_H, int output_W, int K, int *kernel_stride, int *splitxH, int *splitxW, int N) {
    int nH, nW;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bz = tid / (K * nH * nW * B); //n
    int by = (tid % (K * nH * nW * B)) / (K * nH * nW); //b
    int bx = (tid % (K * nH * nW * B) % (K * nH * nW)) / K; //h*w
    int tx = tid % (K * nH * nW * B) % (K * nH * nW) % K; //K
//    int bz = tid / 1024;

    int h = bx / nW; 
    int w = bx % nW;

    if (tid < N) {
    float product_patch[16];
//    product_patch[0] = wino_output[((((0 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[1] = wino_output[((((1 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[2] = wino_output[((((2 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[3] = wino_output[((((3 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[4] = wino_output[((((4 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[6] = wino_output[((((6 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[7] = wino_output[((((7 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[8] = wino_output[((((8 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[9] = wino_output[((((9 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[10] = wino_output[((((10 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[11] = wino_output[((((11 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[12] = wino_output[((((12 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[13] = wino_output[((((13 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[14] = wino_output[((((14 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    product_patch[15] = wino_output[((((15 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//    clock_t time_[9];
//    time_[0] = clock(); 
//    for(int i = 0; i < splitxH[bz]*splitxW[bz]; i++) {
    for(int i = 0; i < 16; i++) {
      product_patch[i] = wino_output[((((i + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
//      product_patch[i] = wino_output[(i + kernel_stride[bz]) * B*nH*nW*K + tid%(K * nH * nW * B)];
//      product_patch[i] = wino_output[(i + kernel_stride[bz]*B*nH*nW*K*16)+tid*16];
    }
//    time_[1] = clock();
//    for(int i = 0; i < 16; i++) {
//      wino_output[tid] = product_patch[i];
//    }
//    time_[2] = clock();

    float output_patch[4];

//    time_[5] = clock(); 

//////// TODO: transformation functions here /////////////
//
//  __device__ function();
    outputWino2NormCalculation2D(product_patch, output_patch, splitxH[bz] - 1, splitxW[bz] - 1);
//
//////////////////////////////////////////////////////////

//    time_[6] = clock();

    tmp_output[(((bz * B + by) * output_H + (2 * h + 0)) * output_W + (2 * w + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || w != nW - 1)
      tmp_output[(((bz * B + by) * output_H + (2 * h + 0)) * output_W + (2 * w + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || h != nH - 1)
      tmp_output[(((bz * B + by) * output_H + (2 * h + 1)) * output_W + (2 * w + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || w != nW - 1) && (output_H % 2 == 0 || h != nH - 1))
      tmp_output[(((bz * B + by) * output_H + (2 * h + 1)) * output_W + (2 * w + 1)) * K + tx] = output_patch[3];

//    if (tid == 1928) {
//        for(int i = 0; i < 2; i++) {
//            time[i] = (int)(time_[i+1] - time_[i]);
//        }
//    }
    }

}

int main() {
    int N;
    int B = 1;
    int K = 32*2;
    int nH = 509;
    int nW = 200;
    int step = 9;

    int splitxH[9] = {4, 4, 4, 4, 4, 4, 2, 2, 2};
    int splitxW[9] = {4, 4, 2, 4, 4, 2, 4, 4, 2};
    int *splitxH_gpu = nullptr;
    int *splitxW_gpu = nullptr;
    cudaMalloc((void**)&splitxH_gpu, step*sizeof(float));
    cudaMalloc((void**)&splitxW_gpu, step*sizeof(float));
    cudaMemcpy(splitxH_gpu, splitxH, step*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(splitxW_gpu, splitxW, step*sizeof(int), cudaMemcpyHostToDevice);

    int kernel_stride[9] = {0, 16, 32, 40, 56, 72, 80, 88, 96};
    int *kernel_stride_gpu = nullptr;
    cudaMalloc((void**)&kernel_stride_gpu, step*sizeof(float));
    cudaMemcpy(kernel_stride_gpu, kernel_stride, step*sizeof(int), cudaMemcpyHostToDevice);
    
    N = K * nH * nW * B * step;
    float *data = nullptr;
    int *time = nullptr;
    cudaMalloc((void**)&data, N*100*sizeof(float));
    cudaMalloc((void**)&time, 9*sizeof(int));

    float *tmp_out = nullptr;
    cudaMalloc((void**)&tmp_out, N*4*step*sizeof(float));
    read <<<(N - 1 + 256) / 256, 256>>> (data, tmp_out, time, B, nH*2, nW*2, K, kernel_stride_gpu, splitxH_gpu, splitxW_gpu, N);
    int time_host[9];
    cudaMemcpy(time_host, time, 9*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++) {
        fprintf(stdout, "input %d:%ld=%f(ms)\n", i,time_host[i], ((float)(time_host[i])/1620000000.0f)*1000.0);
    }

    return 0;
}
