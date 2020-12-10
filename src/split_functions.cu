#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "cuda_fp16.h"
#include <iostream>
#include "src/split_functions.h"
#include "src/split_kernels.cu"

using namespace std;
template <>
void splitFeatureLauncher<float>(const float *Input, float *Splitout, int D_start, int D_end, int H_start, int H_end, int W_start, int W_end, int B, int D, int H ,int W, int C, int padD, int padH, int padW)
{
    //H_start H_end W_start W_end:The coordinates of the extended image
    int D_up=(padD > D_start) ? padD : D_start;
    int D_down=(D+padD < D_end) ? D+padD :D_end;
    int H_up=(padH > H_start) ? padH : H_start;
    int H_down=(H+padH < H_end) ? H+padH :H_end;
    int W_left=(padW > W_start) ? padW : W_start;
    int W_right=(W+padW < W_end) ? W+padW :W_end;
    int D_size=D_end-D_start;
    int H_size=H_end-H_start;
    int W_size=W_end-W_start;
    cudaMemset(Splitout, 0, B*(D_size)*(H_size)*(W_size)*C * sizeof(float));
    dim3 blockDim(C, H_down-H_up, 1);
    dim3 gridDim(W_right-W_left, D_down-D_up, B);
    splitFeatureKernel<float><<<gridDim, blockDim>>>(Input, Splitout, D_start , D_end, H_start, H_end, W_start, W_end, D_size, H_size, W_size, B, D, H, W ,C, D_up, H_up ,W_left, padD, padH, padW);
}

template <>
void splitFilterLauncher<float>(const float *Input, float *Splitout, int D_start, int D_end, int H_start, int H_end, int W_start, int W_end, int D, int H, int W ,int C, int K)
{
    const int D_size=D_end-D_start;
    const int H_size=H_end-H_start;
    const int W_size=W_end-W_start;
    dim3 blockDim(K, H_size, 1);
    dim3 gridDim(W_size, D_size, C);
    cudaMemset(Splitout, 0, D_size*H_size*W_size*C*K*sizeof(float));
    splitFilterKernel<float><<<gridDim, blockDim>>>(Input, Splitout, D_start, H_start, W_start, D_size, H_size, W_size, D, H, W, C, K);
}

template <>
void addFeatureLauncher<float>(float *Result, float *Input1, float *Input2, int B, int D, int H, int W ,int C)
{
    int size = B*D*H*W*C;
    int thread = 512;
    addFeatureKernel<float><<<(size+thread-1)/thread, thread>>>(Result, Input1, Input2, size);
}

template <>
void addFeatureLauncher<at::Half>(at::Half *Result, at::Half *Input1, at::Half *Input2, int B, int D, int H, int W ,int C)
{
    int size = B*D*H*W*C;
    int thread = 512;
    addFeatureKernel<at::Half><<<(size+thread-1)/thread, thread>>>(Result, Input1, Input2, size);
}

template <>
void zeroInit<float>(float *grad, int size)
{cudaMemset(grad, 0, size * sizeof(float));}

template <>
void zeroInit<int>(int *grad, int size)
{cudaMemset(grad, 0, size * sizeof(int));}

template <>
void zeroInit<at::Half>(at::Half *grad, int size)
{cudaMemset(grad, 0, size * sizeof(at::Half));}

