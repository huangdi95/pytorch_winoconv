#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "composer_kernels.cu"

template<unsigned int bn, unsigned int bc, unsigned int bk>
void winograd2DFused(const float *input, const float *tmp_weight_buffer, float *output, float *tmp_out_buffer, const int *kernel_stride, const int *H_start, const int *W_start, int kernel_H, int kernel_W, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split, int output_size) {
    const int N = nH * nW * B;  //N 
    const int maxbytes = 67584; // output smem

    if (kernel_H == 7 and kernel_W == 7) {
//        cudaEvent_t event1;
//        cudaEvent_t event2;
//        cudaEvent_t event3;
//        cudaEvent_t event4;
//        cudaEventCreate(&event1);
//        cudaEventCreate(&event2);
//        cudaEventCreate(&event3);
//        cudaEventCreate(&event4);
//        at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool();
//        at::cuda::CUDAStream stream2 = at::cuda::getStreamFromPool();
//        at::cuda::CUDAStream stream3 = at::cuda::getStreamFromPool();
//        at::cuda::CUDAStream stream4 = at::cuda::getStreamFromPool();
        cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 0, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        cudaFuncSetAttribute(winograd2DFused_8<bn, bc, bk, 3, 1, 4, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        cudaFuncSetAttribute(winograd2DFused_8<bn, bc, bk, 1, 3, 6, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//        cudaFuncSetAttribute(winograd2DFused_4<bn, bc, bk, 1, 1, 8, 9>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        winograd2DFused_16<bn, bc, bk, 3, 3, 0, 4><<<(N/bn)*(K/bk), 256, maxbytes>>>(input, tmp_weight_buffer, tmp_out_buffer, kernel_stride, H_start, W_start, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w, num_split);
//        cudaEventRecord(event1, stream1);
        winograd2DFused_8<bn, bc, bk, 3, 1, 4, 6><<<(N/bn)*(K/bk), 256, maxbytes>>>(input, tmp_weight_buffer, tmp_out_buffer+output_size, kernel_stride, H_start, W_start, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w, num_split);
//        cudaEventRecord(event2, stream2);
        winograd2DFused_8<bn, bc, bk, 1, 3, 6, 8><<<(N/bn)*(K/bk), 256, maxbytes>>>(input, tmp_weight_buffer, tmp_out_buffer+2*output_size, kernel_stride, H_start, W_start, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w, num_split);
//        cudaEventRecord(event3, stream3);
        winograd2DFused_4<bn, bc, bk, 1, 1, 8, 9><<<(N/bn)*(K/bk), 256, maxbytes>>>(input, tmp_weight_buffer, tmp_out_buffer+3*output_size, kernel_stride, H_start, W_start, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w, num_split);
//        cudaEventRecord(event4, stream4);
////        cuda::CUDAStream::waitStream(stream1);
//        cudaStreamWaitEvent(stream1, event1, 0);
//        cudaStreamWaitEvent(stream1, event2, 0);
//        cudaStreamWaitEvent(stream1, event3, 0);
//        cudaStreamWaitEvent(stream1, event4, 0);
        agg <4> <<<(output_size + 256 - 1) / 256, 256>>> (tmp_out_buffer, output, output_size);
    } else if (kernel_H == 9 and kernel_W == 9) {
        cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 0, 9>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        winograd2DFused_16<bn, bc, bk, 3, 3, 0, 9><<<(N/bn)*(K/bk), 256, maxbytes>>>(input, tmp_weight_buffer, output, kernel_stride, H_start, W_start, nH, nW, B, H, W, C, K, output_H, output_W, pad_h, pad_w, num_split);
    } else {
        printf("Error for kernel size (%d, %d)!\n", kernel_H, kernel_W);
        exit(-1);
    }
}
