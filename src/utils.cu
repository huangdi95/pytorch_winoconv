#include <cublas_v2.h>
#include <iostream>
#include "base_conv_launchers.h"

template <typename T>
__global__ void forwardAssign(const T *Input, const T *Weight, T *tmp_data_buffer, const T **Input_ptrs_gpu, const T **Weight_ptrs_gpu, T **tmp_product_ptrs_gpu, int C, int B, int nD, int nH, int nW, int K) {
	int tx = threadIdx.x; // kernel_size
	
	Input_ptrs_gpu[tx] = Input + tx * B * nD * nH * nW * C;
	Weight_ptrs_gpu[tx] = Weight + tx * K * C;
	tmp_product_ptrs_gpu[tx] = tmp_data_buffer + tx * nD * nH * nW * B * K;
}
