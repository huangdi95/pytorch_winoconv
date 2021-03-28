/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Sun 28 Mar 2021 05:43:25 PM CST
 ************************************************************************/
#include <cublas_v2.h>
#include "../base_conv_launchers.h"
#include "../utils.cu"
#include <iostream>
#include <stdio.h>
#include "transform2d.cu"
//time measure
#include <chrono>

//#ifdef CUBLAS_API_H_
//static const char *_cudaGetErrorEnum(cublasStatus_t error)
//{
//    switch (error)
//    {
//        case CUBLAS_STATUS_SUCCESS:
//            return "CUBLAS_STATUS_SUCCESS";
//
//        case CUBLAS_STATUS_NOT_INITIALIZED:
//            return "CUBLAS_STATUS_NOT_INITIALIZED";
//
//        case CUBLAS_STATUS_ALLOC_FAILED:
//            return "CUBLAS_STATUS_ALLOC_FAILED";
//
//        case CUBLAS_STATUS_INVALID_VALUE:
//            return "CUBLAS_STATUS_INVALID_VALUE";
//
//        case CUBLAS_STATUS_ARCH_MISMATCH:
//            return "CUBLAS_STATUS_ARCH_MISMATCH";
//
//        case CUBLAS_STATUS_MAPPING_ERROR:
//            return "CUBLAS_STATUS_MAPPING_ERROR";
//
//        case CUBLAS_STATUS_EXECUTION_FAILED:
//            return "CUBLAS_STATUS_EXECUTION_FAILED";
//
//        case CUBLAS_STATUS_INTERNAL_ERROR:
//            return "CUBLAS_STATUS_INTERNAL_ERROR";
//    }
//
//    return "<unknown>";
//}
//#endif
//#define checkCudaErrors( a ) do { \
//if (cudaSuccess != (a)) { \
//f//printf(stderr, "Cuda runtime error in line %d of file %s \
//: %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
//exit(EXIT_FAILURE); \
//} \
//} while(0);
//#define checkCudaErrors2( a ) do { \
//if (CUBLAS_STATUS_SUCCESS != (a)) { \
//f//printf(stderr, "Cuda runtime error in line %d of file %s \
//: %s \n", __LINE__, __FILE__, _cudaGetErrorEnum(a) ); \
//exit(EXIT_FAILURE); \
//} \
//} while(0);
using namespace std;

template <typename T>
__global__ void outputWino2NormTransform2D_permute(const T *wino_output, T *tmp_output, const int *kernel_stride,  const int *H_start, const int *H_end, const int *W_start, const int *W_end, int B, int output_H, int output_W, int K, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
    int nH, nW;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    int bz = tid / (K * nH * nW * B); //n
    int by = (tid % (K * nH * nW * B)) / (K * nH * nW); //b
    int bx = (tid % (K * nH * nW * B) % (K * nH * nW)) / K; //h*w
    int tx = tid % (K * nH * nW * B) % (K * nH * nW) % K; //K

    int h = bx / nW; 
    int w = bx % nW;

    int splitxH = H_end[bz] - H_start[bz] + 1;
    int splitxW = W_end[bz] - W_start[bz] + 1;

    T product_patch[16] = {0};

    for(int i = 0; i < splitxH*splitxW; i++) {
      product_patch[i] = wino_output[((((i + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    }

    T output_patch[4] = {0};

    outputWino2NormCalculation2D(product_patch, output_patch, splitxH - 1, splitxW - 1);

    tmp_output[(((bz * B + tx) * output_H + (2 * h + 0)) * output_W + (2 * w + 0)) * B + by] = output_patch[0];
    if(output_W % 2 == 0 || w != nW - 1)
      tmp_output[(((bz * B + tx) * output_H + (2 * h + 0)) * output_W + (2 * w + 1)) * B + by] = output_patch[1];
    if(output_H % 2 == 0 || h != nH - 1)
      tmp_output[(((bz * B + tx) * output_H + (2 * h + 1)) * output_W + (2 * w + 0)) * B + by] = output_patch[2];
    if((output_W % 2 == 0 || w != nW - 1) && (output_H % 2 == 0 || h != nH - 1))
      tmp_output[(((bz * B + tx) * output_H + (2 * h + 1)) * output_W + (2 * w + 1)) * B + by] = output_patch[3];
    }
}

template <>
void convLauncherStrideOneLarge2D_base<float>(const float *input, const float *weight,
                              float *tmp_input_buffer, float *tmp_weight_buffer,
                              float *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int B, int H, int W, int C, int K,
                              int kernel_H, int kernel_W, int pad_h, int pad_w,
                              float *output,
                              int num_split, int *H_start_gpu, int *H_end_gpu, int *W_start_gpu, int *W_end_gpu, float *tmp_out_buffer, cublasHandle_t handle)
{
    int output_H = (H + 2 * pad_h - kernel_H) / 1 + 1;
    int output_W = (W + 2 * pad_w - kernel_W) / 1 + 1;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;

//////////////////// a large LUT ///////////////////////
    int num_split2;
    int *H_start = nullptr;
    int *W_start = nullptr;
    int *H_end = nullptr;
    int *W_end = nullptr;
    splitControl2D(kernel_H, kernel_W, &num_split2, &H_start, &H_end, &W_start, &W_end); 

    int *kernel_stride = new int[num_split]();
    for(int i = 1; i < num_split; i++) {
        kernel_stride[i] = (H_end[i-1] - H_start[i-1] + 1) * (W_end[i-1] - W_start[i-1] + 1) + kernel_stride[i-1];
    }
    int kernel_size = kernel_stride[num_split-1] + (H_end[num_split-1] - H_start[num_split-1] + 1) * (W_end[num_split-1] - W_start[num_split-1] + 1);

    int *kernel_stride_gpu = nullptr;
    cudaMalloc((void**)&kernel_stride_gpu, num_split*sizeof(int));
    cudaMemcpy(kernel_stride_gpu, kernel_stride, num_split*sizeof(int), cudaMemcpyHostToDevice);

    int N;
//    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    N = C * nH * nW * B * num_split;
    cout << "N: " << N << endl;
    cout << "kernel_size: " << kernel_size << endl;
    inputNorm2WinoTransform2D2 <float> <<<(N - 1 + 512) / 512, 512>>> (input, tmp_input_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, nH, nW, B, H, W, C, pad_h, pad_w, N);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, num_split, 1);
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (weight, tmp_weight_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, kernel_H, kernel_W, C, K);

    const float** Input_ptrs_gpu_ = (const float **)(tmp_ptr_buffer);
    const float** Weight_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + kernel_size);
    float** tmp_product_ptrs_gpu_ = (float **)(tmp_ptr_buffer + kernel_size*2);

    dim3 bDim3(kernel_size, 1, 1);
    dim3 gDim3(1, 1, 1);
    forwardAssign2D <float> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

    float one = 1;
    float zero = 0;
  
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        &one,
        Weight_ptrs_gpu_, K,
        Input_ptrs_gpu_, C,
        &zero, tmp_product_ptrs_gpu_, K, kernel_size);

    N = num_split*B*nH*nW*K;
    outputWino2NormTransform2D_permute <float> <<<(N - 1 + 512) / 512, 512>>> (tmp_product_buffer, tmp_out_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, B, output_H, output_W, K, N);

    N = B*output_H*output_W*K;
    outputAggregate2D<float> <<<(N - 1 + 512) / 512, 512>>> (tmp_out_buffer, output, num_split, B, output_H, output_W, K, N);

    cudaFree(kernel_stride_gpu);
    delete[] H_start;
    delete[] W_start;
    delete[] H_end;
    delete[] W_end;
    delete[] kernel_stride;
}
