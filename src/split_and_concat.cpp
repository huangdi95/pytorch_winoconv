#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdio.h>
#include <assert.h> 

#include <iostream>
#include <string>

#include "src/split_and_concat_launchers.h"

using namespace std;

template <typename Dtype>
void WinoSplit(at::Tensor in_a, at::Tensor out, int block_D, int block_H, int block_W, int output_block_size, int nD, int nH, int nW, bool two_dimension) {
  if(two_dimension) {
    int B = in_a.sizes()[0];
    int C = in_a.sizes()[3];
    int H = in_a.sizes()[1];
    int W = in_a.sizes()[2];
    int output_block_H = output_block_size;
    int output_block_W = output_block_size;
    int pad_h = 0;
    int pad_w = 0;
    winoSplitLauncher2d(in_a.data<Dtype>(), B, H, W, C, pad_h, pad_w, block_H, block_W, nH, nW, out.data<Dtype>());
  } else {
    int B = in_a.sizes()[0];
    int C = in_a.sizes()[4];
    int D = in_a.sizes()[1];
    int H = in_a.sizes()[2];
    int W = in_a.sizes()[3];
    int output_block_D = output_block_size;
    int output_block_H = output_block_size;
    int output_block_W = output_block_size;
    int pad_d = 0;
    int pad_h = 0;
    int pad_w = 0;
    winoSplitLauncher(in_a.data<Dtype>(), B, D, H, W, C, pad_d, pad_h, pad_w, block_D, block_H, block_W, nD, nH, nW, out.data<Dtype>());
  }
}

template void WinoSplit<float>(at::Tensor in_a, at::Tensor out, int block_D, int block_H, int block_W, int output_block_size, int nD, int nH, int nW, bool two_dimension);

template <typename Dtype>
void WinoConcat(at::Tensor in_a, at::Tensor out, int output_block_size, bool two_dimension) {
  if(two_dimension) {
    int N = in_a.sizes()[0];
    int K = in_a.sizes()[1];
    int output_block_H = output_block_size;
    int output_block_W = output_block_size;
    int output_H = out.sizes()[1];
    int output_W = out.sizes()[2];
    int nH = (output_H+output_block_H-1)/output_block_H;
    int nW = (output_W+output_block_W-1)/output_block_W;
    int B = N / (nH * nW);

    winoConcatLauncher2d(in_a.data<Dtype>(), B, output_H, output_W, K, output_block_H, output_block_W, out.data<Dtype>());
    out.resize_({B, output_H, output_W, K});
  } else {
    int N = in_a.sizes()[0];
    int K = in_a.sizes()[1];
    int output_block_D = output_block_size;
    int output_block_H = output_block_size;
    int output_block_W = output_block_size;
    int output_D = out.sizes()[1];
    int output_H = out.sizes()[2];
    int output_W = out.sizes()[3];
    int nD = (output_D+output_block_D-1)/output_block_D;
    int nH = (output_H+output_block_H-1)/output_block_H;
    int nW = (output_W+output_block_W-1)/output_block_W;
    int B = N / (nH * nW * nD);

    winoConcatLauncher(in_a.data<Dtype>(), B, output_D, output_H, output_W, K, output_block_D, output_block_H, output_block_W, out.data<Dtype>());
    out.resize_({B, output_D, output_H, output_W, K});
  }
}

template void WinoConcat<float>(at::Tensor in_a, at::Tensor out, int output_block_size, bool two_dimension);
