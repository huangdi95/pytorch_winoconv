#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include "src/base_conv_launchers.h"
#include "src/utils.cu"
#include <iostream>
//#include "utils.cu.cc"
#include <stdio.h>
#include "transform.cu"
#include "transform2d.cu"
#ifdef CUBLAS_API_H_
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif
#define checkCudaErrors( a ) do { \
if (cudaSuccess != (a)) { \
fprintf(stderr, "Cuda runtime error in line %d of file %s \
: %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
exit(EXIT_FAILURE); \
} \
} while(0);
#define checkCudaErrors2( a ) do { \
if (CUBLAS_STATUS_SUCCESS != (a)) { \
fprintf(stderr, "Cuda runtime error in line %d of file %s \
: %s \n", __LINE__, __FILE__, _cudaGetErrorEnum(a) ); \
exit(EXIT_FAILURE); \
} \
} while(0);
//#include "333.cu.cc"
//#include "333_2.cu.cc"
using namespace std;

template <>
void convLauncherStrideOneLarge<float>(const float *input, const float *weight,
                              float *tmp_input_buffer, float *tmp_weight_buffer,
                              float *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int B, int D, int H, int W, int C, int K,
                              int kernel_D, int kernel_H, int kernel_W, int pad_d, int pad_h, int pad_w,
                              float *output)
{
    int output_D = (D + 2 * pad_d - kernel_D) / 1 + 1;
    int output_H = (H + 2 * pad_h - kernel_H) / 1 + 1;
    int output_W = (W + 2 * pad_w - kernel_W) / 1 + 1;
    int nD = (output_D + 1) / 2;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;

//////////////////// a large LUT ///////////////////////
    int num_split;
    int *D_start = nullptr;
    int *H_start = nullptr;
    int *W_start = nullptr;
    int *D_end = nullptr;
    int *H_end = nullptr;
    int *W_end = nullptr;
    splitControl(kernel_D, kernel_H, kernel_W, &num_split, &D_start, &D_end, &H_start, &H_end, &W_start, &W_end); 

    int *kernel_stride = new int[num_split]();
    for(int i = 1; i < num_split; i++) {
        kernel_stride[i] = (D_end[i-1] - D_start[i-1] + 1) * (H_end[i-1] - H_start[i-1] + 1) * (W_end[i-1] - W_start[i-1] + 1) + kernel_stride[i-1];
//        cout << kernel_stride[i] << endl;
    }

    int kernel_size = kernel_stride[num_split-1] + (D_end[num_split-1] - D_start[num_split-1] + 1) * (H_end[num_split-1] - H_start[num_split-1] + 1) * (W_end[num_split-1] - W_start[num_split-1] + 1);
//    cout << "kernel_size: " << kernel_size << endl;

    int *D_start_gpu = nullptr;
    int *H_start_gpu = nullptr;
    int *W_start_gpu = nullptr;
    int *D_end_gpu = nullptr;
    int *H_end_gpu = nullptr;
    int *W_end_gpu = nullptr;
    int *kernel_stride_gpu = nullptr;
    cudaMalloc((void**)&D_start_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&D_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&H_start_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&H_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&W_start_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&W_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&kernel_stride_gpu, num_split*sizeof(int));
    cudaMemcpy(D_start_gpu, D_start, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(H_start_gpu, H_start, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(W_start_gpu, W_start, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D_end_gpu, D_end, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(H_end_gpu, H_end, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(W_end_gpu, W_end, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_stride_gpu, kernel_stride, num_split*sizeof(int), cudaMemcpyHostToDevice);

    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nD*nH*nW, B, num_split);
    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, num_split, 1);
//    cout << "---------------------" << endl;
//    cout << B << D << H << W << C << K << num_split << endl;
//    cout << output_D << " " << output_H << " " << output_W << endl;
//    cout << "---------------------" << endl;
    inputNorm2WinoTransform <float> <<<gDim1, bDim1>>> (input, tmp_input_buffer, kernel_stride_gpu, D_start_gpu, D_end_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, nD, nH, nW, B, D, H, W, C, pad_d, pad_h, pad_w);
    wNorm2WinoTransform <float> <<<gDim2, bDim2>>> (weight, tmp_weight_buffer, kernel_stride_gpu, D_start_gpu, D_end_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, kernel_D, kernel_H, kernel_W, C, K);

    const float** Input_ptrs_gpu_ = (const float **)(tmp_ptr_buffer);
    const float** Weight_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + kernel_size);
    float** tmp_product_ptrs_gpu_ = (float **)(tmp_ptr_buffer + kernel_size * 2);

    dim3 bDim3(kernel_size, 1, 1);
    dim3 gDim3(1, 1, 1);
    forwardAssign <float> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nD, nH, nW, K);

    float one = 1;
    float zero = 0;
  
    cublasHandle_t handle;
    checkCudaErrors2(cublasCreate(&handle));
    checkCudaErrors2(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nD * nH * nW, C,
        &one,
        Weight_ptrs_gpu_, K,
        Input_ptrs_gpu_, C,
        &zero, tmp_product_ptrs_gpu_, K, kernel_size));

//    cout << K << endl;
//    cout << C << endl;
//    cout << B * nD * nH * nW << endl;
//
    dim3 bDim4(K, 1, 1);
    dim3 gDim4(nD*nH*nW, B, num_split);
    float *tmp_output = nullptr;
    checkCudaErrors(cudaMalloc((void**)&tmp_output, num_split*B*output_D*output_H*output_W*K*sizeof(float)));
    outputWino2NormTransform <float> <<<gDim4, bDim4>>> (tmp_product_buffer, tmp_output, kernel_stride_gpu, D_start_gpu, D_end_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, B, output_D, output_H, output_W, K);

    dim3 bDim5(K, 1, 1);
    dim3 gDim5(output_D*output_H*output_W, B, 1);
    outputAggregate<float> <<<gDim5, bDim5>>> (tmp_output, output, num_split, B, output_D, output_H, output_W, K);

    checkCudaErrors2(cublasDestroy(handle));
    cudaFree(D_start_gpu);
    cudaFree(H_start_gpu);
    cudaFree(W_start_gpu);
    cudaFree(D_end_gpu);
    cudaFree(H_end_gpu);
    cudaFree(W_end_gpu);
    cudaFree(kernel_stride_gpu);
    cudaFree(tmp_output);
    delete[] D_start;
    delete[] H_start;
    delete[] W_start;
    delete[] D_end;
    delete[] H_end;
    delete[] W_end;
    delete[] kernel_stride;
}

template <>
void convLauncherStrideOneLarge2D<float>(const float *input, const float *weight,
                              float *tmp_input_buffer, float *tmp_weight_buffer,
                              float *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int B, int H, int W, int C, int K,
                              int kernel_H, int kernel_W, int pad_h, int pad_w,
                              float *output)
{
    int output_H = (H + 2 * pad_h - kernel_H) / 1 + 1;
    int output_W = (W + 2 * pad_w - kernel_W) / 1 + 1;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;

//////////////////// a large LUT ///////////////////////
    int num_split;
    int *H_start = nullptr;
    int *W_start = nullptr;
    int *H_end = nullptr;
    int *W_end = nullptr;
    splitControl2D(kernel_H, kernel_W, &num_split, &H_start, &H_end, &W_start, &W_end); 

    int *kernel_stride = new int[num_split]();
    for(int i = 1; i < num_split; i++) {
        kernel_stride[i] = (H_end[i-1] - H_start[i-1] + 1) * (W_end[i-1] - W_start[i-1] + 1) + kernel_stride[i-1];
//        cout << kernel_stride[i] << endl;
    }

    int kernel_size = kernel_stride[num_split-1] + (H_end[num_split-1] - H_start[num_split-1] + 1) * (W_end[num_split-1] - W_start[num_split-1] + 1);
//    cout << "kernel_size: " << kernel_size << endl;

    int *H_start_gpu = nullptr;
    int *W_start_gpu = nullptr;
    int *H_end_gpu = nullptr;
    int *W_end_gpu = nullptr;
    int *kernel_stride_gpu = nullptr;
    cudaMalloc((void**)&H_start_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&H_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&W_start_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&W_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&kernel_stride_gpu, num_split*sizeof(int));
    cudaMemcpy(H_start_gpu, H_start, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(W_start_gpu, W_start, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(H_end_gpu, H_end, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(W_end_gpu, W_end, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_stride_gpu, kernel_stride, num_split*sizeof(int), cudaMemcpyHostToDevice);

    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nH*nW, B, num_split);
    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, num_split, 1);
    inputNorm2WinoTransform2D <float> <<<gDim1, bDim1>>> (input, tmp_input_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, nH, nW, B, H, W, C, pad_h, pad_w);
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (weight, tmp_weight_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, kernel_H, kernel_W, C, K);

    const float** Input_ptrs_gpu_ = (const float **)(tmp_ptr_buffer);
    const float** Weight_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + kernel_size);
    float** tmp_product_ptrs_gpu_ = (float **)(tmp_ptr_buffer + kernel_size * 2);

    dim3 bDim3(kernel_size, 1, 1);
    dim3 gDim3(1, 1, 1);
    forwardAssign2D <float> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

    float one = 1;
    float zero = 0;
  
    cublasHandle_t handle;
    checkCudaErrors2(cublasCreate(&handle));
    checkCudaErrors2(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        &one,
        Weight_ptrs_gpu_, K,
        Input_ptrs_gpu_, C,
        &zero, tmp_product_ptrs_gpu_, K, kernel_size));

    dim3 bDim4(K, 1, 1);
    dim3 gDim4(nH*nW, B, num_split);
    float *tmp_output = nullptr;
    checkCudaErrors(cudaMalloc((void**)&tmp_output, num_split*B*output_H*output_W*K*sizeof(float)));
    outputWino2NormTransform2D <float> <<<gDim4, bDim4>>> (tmp_product_buffer, tmp_output, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, B, output_H, output_W, K);

    dim3 bDim5(K, 1, 1);
    dim3 gDim5(output_H*output_W, B, 1);
    outputAggregate2D<float> <<<gDim5, bDim5>>> (tmp_output, output, num_split, B, output_H, output_W, K);

    checkCudaErrors2(cublasDestroy(handle));
    cudaFree(H_start_gpu);
    cudaFree(W_start_gpu);
    cudaFree(H_end_gpu);
    cudaFree(W_end_gpu);
    cudaFree(kernel_stride_gpu);
    cudaFree(tmp_output);
    delete[] H_start;
    delete[] W_start;
    delete[] H_end;
    delete[] W_end;
    delete[] kernel_stride;
}
