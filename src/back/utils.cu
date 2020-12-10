#include <cublas_v2.h>
#include <iostream>

using namespace std;

template <typename T>
__global__ void forwardAssign(const T *Input, const T *Weight, T *tmp_data_buffer, const T **Input_ptrs_gpu, const T **Weight_ptrs_gpu, T **tmp_product_ptrs_gpu, int C, int B, int nH, int nW, int K) {
	int tx = threadIdx.x; // kernel_size
	
	Input_ptrs_gpu[tx] = Input + tx * B * nH * nW * C;
	Weight_ptrs_gpu[tx] = Weight + tx * K * C;
	tmp_product_ptrs_gpu[tx] = tmp_data_buffer + tx * nH * nW * B * K;
}

template <typename T>
__global__ void backwardAssign(const float *tmp_input_buffer, const float *tmp_weight_buffer, float *tmp_bottom_diff_buffer, float *tmp_weight_diff_buffer, float *tmp_top_diff_buffer, const float** tmp_input_buffer_ptrs_gpu, float** tmp_bottom_diff_buffer_ptrs_gpu, const float** tmp_weight_buffer_ptrs_gpu, float** tmp_weight_diff_buffer_ptrs_gpu, const float** tmp_top_diff_buffer_ptrs_gpu, int C, int B, int nH, int nW, int K) {
	int tx = threadIdx.x; // kernel_size

	tmp_input_buffer_ptrs_gpu [tx] = tmp_input_buffer + tx * B * nH * nW * C;
	tmp_bottom_diff_buffer_ptrs_gpu [tx] = tmp_bottom_diff_buffer + tx * B * nH * nW * C;
	tmp_weight_buffer_ptrs_gpu [tx] = tmp_weight_buffer + tx * K * C;
	tmp_weight_diff_buffer_ptrs_gpu [tx] = tmp_weight_diff_buffer + tx * K * C;
	tmp_top_diff_buffer_ptrs_gpu [tx] = tmp_top_diff_buffer + tx * nH * nW * B * K;
}
