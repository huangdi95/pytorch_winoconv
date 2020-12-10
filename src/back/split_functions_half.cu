#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include "cuda_fp16.h"
#include <iostream>
#include "src/split_functions.h"
#include "src/split_kernels.cu"

using namespace std;

template <>
void splitFeatureLauncher<at::Half>(const at::Half *Input, at::Half *Splitout, int H_start, int H_end, int W_start, int W_end, int B, int H ,int W, int C,int padH, int padW)
{
  //H_start H_end W_start W_end:The coordinates of the extended image
  int H_up=(padH > H_start) ? padH : H_start;
  int H_down=(H+padH < H_end) ? H+padH :H_end;
  int W_left=(padW > W_start) ? padW : W_start;
  int W_right=(W+padW < W_end) ? W+padW :W_end;
  int H_size=H_end-H_start;
  int W_size=W_end-W_start;
  cudaMemset(Splitout, 0, B*(H_size)*(W_size)*C * sizeof(at::Half));
  dim3 blockDim(C, 1, 1);
  dim3 gridDim(B, H_down-H_up, W_right-W_left);
  splitFeatureKernel<at::Half><<<gridDim, blockDim>>>(Input, Splitout, H_start , H_end , W_start, W_end,  H_size, W_size, B, H, W ,C, H_up ,W_left, padH, padW);
}

template <>
void splitFilterLauncher<at::Half>(const at::Half *Input, at::Half *Splitout, int H_start, int H_end, int W_start, int W_end, int H, int W ,int C, int K)
{
  const int H_size=H_end-H_start;
  const int W_size=W_end-W_start;
  dim3 blockDim(K, 1, 1);
  dim3 gridDim(H_size, W_size, C);
  cudaMemset(Splitout, 0, (H_size)*(W_size)*C *K* sizeof(at::Half));
  splitFilterKernel<at::Half><<<gridDim, blockDim>>>(Input, Splitout, H_start , H_end , W_start, W_end,  H_size, W_size, H, W, C, K);
}

template <>
void addFeatureLauncher<at::Half>(at::Half *Result, at::Half *Input1, at::Half *Input2, int B, int H, int W ,int C)
{
  dim3 blockDim(C, 1, 1);
  dim3 gridDim(B, H, W);
  addFeatureKernel<at::Half><<<gridDim, blockDim>>>(Result, Input1, Input2, B, H, W, C);
}

template <>
void zeroInit<at::Half>(at::Half *grad, int size)
{cudaMemset(grad, 0, size * sizeof(at::Half));}

