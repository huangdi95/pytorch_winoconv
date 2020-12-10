#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include "cuda_fp16.h"
#include "src/base_conv_launchers.h"
#include "src/transform_kernels.cu"
#include "src/utils.cu"
#include <iostream>
#include <stdio.h>

using namespace std;

template <>
void convLauncherStrideOne1x1<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 4;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform1x1 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform1x1 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 1) / 1 + 1;
    int output_W = (W + 2 * pad_w - 1) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform1x1 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);

}

template <>
void convLauncherStrideOne1x2<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 6;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform1x2 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform1x2 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 1) / 1 + 1;
    int output_W = (W + 2 * pad_w - 2) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform1x2 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}

template <>
void convLauncherStrideOne1x3<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 8;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform1x3 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform1x3 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 1) / 1 + 1;
    int output_W = (W + 2 * pad_w - 3) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform1x3 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}

template <>
void convLauncherStrideOne2x1<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 6;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform2x1 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform2x1 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 2) / 1 + 1;
    int output_W = (W + 2 * pad_w - 1) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform2x1 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}

template <>
void convLauncherStrideOne2x2<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 9;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform2x2 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform2x2 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 2) / 1 + 1;
    int output_W = (W + 2 * pad_w - 2) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform2x2 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}

template <>
void convLauncherStrideOne2x3<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 12;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform2x3 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform2x3 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 2) / 1 + 1;
    int output_W = (W + 2 * pad_w - 3) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform2x3 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}

template <>
void convLauncherStrideOne3x1<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 8;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform3x1 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform3x1 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 3) / 1 + 1;
    int output_W = (W + 2 * pad_w - 1) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform3x1 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}

template <>
void convLauncherStrideOne3x2<at::Half>(const at::Half *input, const at::Half *weight, 
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer, 
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 12;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform3x2 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform3x2 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 3) / 1 + 1;
    int output_W = (W + 2 * pad_w - 2) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform3x2 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}

template <>
void convLauncherStrideOne3x3<at::Half>(const at::Half *input, const at::Half *weight,
                              at::Half *tmp_input_buffer, at::Half *tmp_weight_buffer,
                              at::Half *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              at::Half *output)
{
    int kernel_size = 16;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    dim3 bDim1(C, 1, 1);
    dim3 gDim1(nW, nH, B);
    inputNorm2WinoTransform3x3 <at::Half> <<<gDim1, bDim1>>> (input, C, B, H, W, pad_h, pad_w, tmp_input_buffer);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, 1, 1);
    wNorm2WinoTransform3x3 <at::Half> <<<gDim2, bDim2>>> (weight, C, K, tmp_weight_buffer);

	const at::Half** Input_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer);
	const at::Half** Weight_ptrs_gpu_ = (const at::Half **)(tmp_ptr_buffer + kernel_size);
	at::Half** tmp_product_ptrs_gpu_ = (at::Half **)(tmp_ptr_buffer + kernel_size * 2);

	dim3 bDim3(kernel_size, 1, 1);
	dim3 gDim3(1, 1, 1);
	forwardAssign <at::Half> <<<gDim3, bDim3>>> (tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

	at::Half one = at::Half(1);
	at::Half zero = at::Half(0);
    
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        (const __half*)(&one),
        (const __half**)(Weight_ptrs_gpu_), K,
        (const __half**)(Input_ptrs_gpu_), C,
        (const __half*)(&zero), (__half**)(tmp_product_ptrs_gpu_), K, kernel_size);

    int output_H = (H + 2 * pad_h - 3) / 1 + 1;
    int output_W = (W + 2 * pad_w - 3) / 1 + 1;
	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	outputWino2NormTransform3x3 <at::Half> <<<gridDim, blockDim>>> (tmp_product_buffer, B, output_H, output_W, K, output);

	cublasDestroy(handle);
}
