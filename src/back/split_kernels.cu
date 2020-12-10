#include <cublas_v2.h>

template <typename T>
__global__ void splitFeatureKernel(
    const T* Input, T* Output,
    int H_start, int H_end, int W_start, int W_end,
    int H_size, int W_size ,int B, int H ,int W, int C, int H_up, int W_left, int padH ,int padW) {
    int b = blockIdx.x; 
    int h = blockIdx.y + H_up - H_start; 
    int w = blockIdx.z + W_left - W_start; 
    int c = threadIdx.x; 
    int out_idx = c + C * (w + W_size * (h + H_size * b));
    Output[out_idx]=Input[c + C * (w + W_start-padW + W * (h + H_start - padH + H * b))];
}

template <typename T>
__global__ void addFeatureKernel(
    T* Result, T* Input1, T *Input2,
int B, int H, int W, int C) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int c = threadIdx.x;
    int idx = c + C * (w + W * (h + H * b));

    Result[idx]=Input1[idx]+Input2[idx];
}

template <typename T>
__global__ void splitFilterKernel(
    const T* Input, T* Output,
    int H_start, int H_end, int W_start, int W_end,
    int H_size, int W_size ,int H, int W ,int C, int K) {
    int h = blockIdx.x;
    int w = blockIdx.y;
    int c = blockIdx.z;
    int k = threadIdx.x;
    int out_idx = k + K * (c + C * (w + W_size * h));
    Output[out_idx]=Input[k + K * (c  + C * (w + W_start + W * (h+H_start)))];
}

template <typename T>
__global__ void splitFilterGradKernel(
    const T* Output_grad, T* Input_grad,
    int H_start, int H_end, int W_start, int W_end,
    int H_size, int W_size ,int H, int W ,int C, int K) {
    int h = blockIdx.x;
    int w = blockIdx.y;
    int c = blockIdx.z;
    int k = threadIdx.x;
    int out_idx = k + K * (c + C * (w + W_size * h));
    Input_grad[k + K * (c  + C * (w + W_start + W * (h+H_start)))]+=Output_grad[out_idx];
}

template <typename T>
__global__ void splitFeatureGradKernel(
    const T* Output_grad, T* Input_grad,
    int H_start, int H_end, int W_start, int W_end,
    int H_size, int W_size ,int B, int H ,int W, int C, int H_up, int W_left, int padH ,int padW) {
    int b = blockIdx.x; 
    int h = blockIdx.y + H_up - H_start; 
    int w = blockIdx.z + W_left - W_start; 
    int c = threadIdx.x; 
    int out_idx = c + C * (w + W_size * (h + H_size * b));
    Input_grad[c + C * (w + W_start-padW + W * (h + H_start - padH + H * b))]+=Output_grad[out_idx];
}

