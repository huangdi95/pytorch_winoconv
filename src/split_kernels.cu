#include <cublas_v2.h>

template <typename T>
__global__ void splitFeatureKernel2(
    const T* Input, T* Output,
    int D_start, int D_end, int H_start, int H_end, int W_start, int W_end,
    int D_size, int H_size, int W_size ,int B, int D, int H ,int W, int C,
    int D_up, int H_up, int W_left, int padD, int padH ,int padW) {
    int b = blockIdx.z; // b
    int d = blockIdx.y + D_up - D_start; // d
    int w = blockIdx.x + W_left - W_start; // w
    int h = threadIdx.y + H_up - H_start; // h
    int c = threadIdx.x; // c

    int out_idx = (((b * D_size + d) * H_size+ h) * W_size + w) * C + c;
    Output[out_idx]=Input[(((b * D + d + D_start - padD) * H + h + H_start - padH) * W + w + W_start - padW) * C + c];
}

template <typename T>
__global__ void splitFeatureKernel(
    const T* Input, T* Output,
    int D_start, int D_end, int H_start, int H_end, int W_start, int W_end,
    int D_size, int H_size, int W_size ,int B, int D, int H ,int W, int C,
    int D_up, int H_up, int W_left, int padD, int padH ,int padW) {
    int b = blockIdx.z; // b
    int d = blockIdx.y + D_up - D_start; // d
    int w = blockIdx.x + W_left - W_start; // w
    int h = threadIdx.y + H_up - H_start; // h
    int c = threadIdx.x; // c

    int out_idx = (((b * D_size + d) * H_size+ h) * W_size + w) * C + c;
    Output[out_idx]=Input[(((b * D + d + D_start - padD) * H + h + H_start - padH) * W + w + W_start - padW) * C + c];
}

template <typename T>
__global__ void addFeatureKernel(T* Result, T* Input1, T *Input2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        Result[idx]=Input1[idx]+Input2[idx];
    }
}

template <typename T>
__global__ void splitFilterKernel(
    const T* Input, T* Output,
    int D_start, int H_start, int W_start,
    int D_size, int H_size, int W_size,
    int D, int H, int W, int C, int K) {
    int c = blockIdx.z; // c
    int d = blockIdx.y; // d
    int w = blockIdx.x; // w
    int h = threadIdx.y; // h
    int k = threadIdx.x; // k
    int out_idx = (((d * H_size + h) * W_size + w) * C + c) * K + k;
    Output[out_idx]=Input[((((d + D_start) * H + h + H_start) * W + w + W_start) * C + c) * K + k];
}

