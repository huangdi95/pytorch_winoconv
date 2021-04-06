#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "src/fused_wino_launchers.h"
#include <iostream>
#include <stdio.h>
#include "transform.cu"
#include "transform2d.cu"
#include "fused_winoconv.cu"
//time measure
#include <chrono>

template <>
void fusedLauncherStrideOne2D<float>(
    const float *input,
    const float *weight,
    float *output,
    float *tmp_weight_buffer,
    int B, int H, int W, int C, int K, int kernel_H, int kernel_W, int pad_h, int pad_w,
    int num_split,
    int *H_start_gpu,
    int *H_end_gpu,
    int *W_start_gpu,
    int *W_end_gpu,
    int *kernel_stride_gpu,
    float *tmp_out_buffer)
{
    int output_H = (H + 2 * pad_h - kernel_H) / 1 + 1;
    int output_W = (W + 2 * pad_w - kernel_W) / 1 + 1;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;
    int output_size = B * output_H * output_W * K;

//    int H_s[] = {0, 0, 3, 3, 0, 3, 6, 6, 6};
//    int H_e[] = {3, 3, 6, 6, 3, 6, 7, 7, 7};
//    int W_s[] = {0, 3, 0, 3, 6, 6, 0, 3, 6};
//    int W_e[] = {3, 6, 3, 6, 7, 7, 3, 6, 7};
//
//    int kernel_stride[num_split] = {0};
//    for(int i = 1; i < num_split; i++) {
//        kernel_stride[i] = (H_e[i-1] - H_s[i-1] + 1) * (W_e[i-1] - W_s[i-1] + 1) + kernel_stride[i-1];
//    }
//    int Batch = kernel_stride[num_split-1] + (H_e[num_split-1] - H_s[num_split-1] + 1) * (W_e[num_split-1] - W_s[num_split-1] + 1);
//
//    float flop = 2 * (float)B * (float)nH * (float)nW *(float)C * (float)K * Batch;
//    printf("wino flop: %f\n", flop);
//    printf("num_split: %d\n", num_split);

//    int *H_start_gpu = nullptr;
//    int *W_start_gpu = nullptr;
//    int *H_end_gpu = nullptr;
//    int *W_end_gpu = nullptr;
//    int *kernel_stride_gpu = nullptr;
//    cudaMalloc((void**)&H_start_gpu, num_split*sizeof(int));
//    cudaMalloc((void**)&H_end_gpu, num_split*sizeof(int));
//    cudaMalloc((void**)&W_start_gpu, num_split*sizeof(int));
//    cudaMalloc((void**)&W_end_gpu, num_split*sizeof(int));
//    cudaMalloc((void**)&kernel_stride_gpu, num_split*sizeof(int));
//    cudaMemcpy(H_start_gpu, H_s, num_split*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(W_start_gpu, W_s, num_split*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(H_end_gpu, H_e, num_split*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(W_end_gpu, W_e, num_split*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(kernel_stride_gpu, kernel_stride, num_split*sizeof(int), cudaMemcpyHostToDevice);

//////////////////// a large LUT ///////////////////////
//    int num_split2;
//    int *H_start = nullptr;
//    int *W_start = nullptr;
//    int *H_end = nullptr;
//    int *W_end = nullptr;
//    splitControlFused2D(kernel_H, kernel_W, &num_split2, &H_start, &H_end, &W_start, &W_end); 
//
//    int *kernel_stride = new int[num_split]();
//    for(int i = 1; i < num_split; i++) {
//        kernel_stride[i] = (H_end[i-1] - H_start[i-1] + 1) * (W_end[i-1] - W_start[i-1] + 1) + kernel_stride[i-1];
//    }
//    int *kernel_stride_gpu = nullptr;
//    cudaMalloc((void**)&kernel_stride_gpu, num_split*sizeof(int));
//    cudaMemcpy(kernel_stride_gpu, kernel_stride, num_split*sizeof(int), cudaMemcpyHostToDevice);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, num_split, 1);
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (weight, tmp_weight_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, kernel_H, kernel_W, C, K);

    if (C % 8 == 0 && K % 64 == 0) {
        winograd2DFused<32, 8, 64>(
            input,
            tmp_weight_buffer,
            output,
            tmp_out_buffer,
            kernel_stride_gpu,
            H_start_gpu,
            W_start_gpu,
            kernel_H, kernel_W, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w,
            num_split,
            output_size);
    } else if (C % 8 == 0 && K % 32 == 0) {
        winograd2DFused<32, 8, 32>(
            input,
            tmp_weight_buffer,
            output,
            tmp_out_buffer,
            kernel_stride_gpu,
            H_start_gpu,
            W_start_gpu,
            kernel_H, kernel_W, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w,
            num_split,
            output_size);
    } else if (C % 7 == 0 && K % 64 == 0) {
        winograd2DFused<32, 7, 64>(
            input,
            tmp_weight_buffer,
            output,
            tmp_out_buffer,
            kernel_stride_gpu,
            H_start_gpu,
            W_start_gpu,
            kernel_H, kernel_W, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w,
            num_split,
            output_size);
    } else if (C % 7 == 0 && K % 32 == 0) {
        winograd2DFused<32, 7, 32>(
            input,
            tmp_weight_buffer,
            output,
            tmp_out_buffer,
            kernel_stride_gpu,
            H_start_gpu,
            W_start_gpu,
            kernel_H, kernel_W, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w,
            num_split,
            output_size);
    } else if (C % 6 == 0 && K % 64 == 0) {
        winograd2DFused<32, 6, 64>(
            input,
            tmp_weight_buffer,
            output,
            tmp_out_buffer,
            kernel_stride_gpu,
            H_start_gpu,
            W_start_gpu,
            kernel_H, kernel_W, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w,
            num_split,
            output_size);
    } else if (C % 6 == 0 && K % 32 == 0) {
        winograd2DFused<32, 6, 32>(
            input,
            tmp_weight_buffer,
            output,
            tmp_out_buffer,
            kernel_stride_gpu,
            H_start_gpu,
            W_start_gpu,
            kernel_H, kernel_W, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w,
            num_split,
            output_size);
    } else {
        printf("Error for channel size (%d, %d)!\n", C, K);
        exit(-1);
    }

//    delete[] H_start;
//    delete[] W_start;
//    delete[] H_end;
//    delete[] W_end;
//    delete[] kernel_stride;
}
