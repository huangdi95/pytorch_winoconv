#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdio.h>
#include <assert.h> 

#include <iostream>
#include <string>

#include "src/split_and_concat_launchers.h"

using namespace std;

template <typename Dtype>
void WinoSplit(at::Tensor in_a, at::Tensor out, int block_H, int block_W, int output_block_size, int nH, int nW) {
  int B = in_a.sizes()[0];
  int C = in_a.sizes()[1];
  int H = in_a.sizes()[2];
  int W = in_a.sizes()[3];
  int output_block_H = output_block_size;
  int output_block_W = output_block_size;
  int pad_h = 0;
  int pad_w = 0;
//  cout << "B: " << B << endl;
//  cout << "C: " << C << endl;
//  cout << "H: " << H << endl;
//  cout << "W: " << W << endl;
//  cout << "nH: " << nH << endl;
//  cout << "nW: " << nW << endl;
//  cout << "output_block_H: " << output_block_H << endl;
//  cout << "output_block_W: " << output_block_W << endl;
  winoSplitLauncher(in_a.data<Dtype>(), B, H, W, C, pad_h, pad_w, block_H, block_W, nH, nW, out.data<Dtype>());
//  out.resize_({B*nH*nW, C, block_H, block_W});
}

template void WinoSplit<float>(at::Tensor in_a, at::Tensor out, int block_H, int block_W, int output_block_size, int nH, int nW);

template <typename Dtype>
void WinoConcat(at::Tensor in_a, at::Tensor out, int output_block_size) {
  int N = in_a.sizes()[0];
  int K = in_a.sizes()[1];
  int output_block_H = output_block_size;
  int output_block_W = output_block_size;
  int output_H = out.sizes()[2];
  int output_W = out.sizes()[3];
  int nH = (output_H+output_block_H-1)/output_block_H;
  int nW = (output_W+output_block_W-1)/output_block_W;
  int B = N / (nH * nW);

  winoConcatLauncher(in_a.data<Dtype>(), B, output_H, output_W, K, output_block_H, output_block_W, out.data<Dtype>());
  out.resize_({B, K, output_H, output_W});
}

template void WinoConcat<float>(at::Tensor in_a, at::Tensor out, int output_block_size);
