#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdio.h>
#include <assert.h> 
#include <math.h>

#include <iostream>
#include <string>
#include "src/base_conv_launchers.h"
#include "src/split_functions.h"
#include "src/split_and_concat_launchers.h"

using namespace std;

template <typename T>
void DWM(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
         at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer
        ) {
  int B = Input.sizes()[0];
  int D = Input.sizes()[1];
  int H = Input.sizes()[2];
  int W = Input.sizes()[3];
  int C = Input.sizes()[4];
  int kernel_D = Weight.sizes()[0];
  int kernel_H = Weight.sizes()[1];
  int kernel_W = Weight.sizes()[2];
  int K = Weight.sizes()[4];

  string padding_ = "VALID" ;
  int padD1 = 0;
  int padD2 = 0;
  int padH1 = 0;
  int padH2 = 0;
  int padW1 = 0;
  int padW2 = 0;
  if (padding_ == "SAME") {
    if(stride == 1) {
      padD1 = (kernel_D - 1) / 2;
      padD2 = (kernel_D) / 2;
      padH1 = (kernel_H - 1) / 2;
      padH2 = (kernel_H) / 2;
      padW1 = (kernel_W - 1) / 2;
      padW2 = (kernel_W) / 2;
    } else if(stride == 2) {
      padD1 = (kernel_D - 2 + D%2) / 2;
      padD2 = (kernel_D - 1 + D%2) / 2;
      padH1 = (kernel_H - 2 + H%2) / 2;
      padH2 = (kernel_H - 1 + H%2) / 2;
      padW1 = (kernel_W - 2 + W%2) / 2;
      padW2 = (kernel_W - 1 + W%2) / 2;
    }
  } else if (padding_ == "VALID"){
    if(stride == 1) {
      padD1 = 0;
      padD2 = 0;
      padH1 = 0;
      padH2 = 0;
      padW1 = 0;
      padW2 = 0;
    } else if (stride == 2){
      padD1 = 0;
      padD2 = -(D+kernel_D)%2;
      padH1 = 0;
      padH2 = -(H+kernel_H)%2;
      padW1 = 0;
      padW2 = -(W+kernel_W)%2;
    }
  }
  
  int output_D = (D + padD1 + padD2 - kernel_D) / stride + 1;
  int output_H = (H + padH1 + padH2 - kernel_H) / stride + 1;
  int output_W = (W + padW1 + padW2 - kernel_W) / stride + 1;

  if(stride == 1) {
//  if(kernel_H == 5 && kernel_W == 5) {
//    pad_h = 2;
//    pad_w = 2;
//  }
    if (padD1 != padD2 or padH1 != padH2 or padW1 != padW2) {
      cout << "padding == " << padding_ << ", kernel_size == (" << kernel_D << ", " << kernel_H << ", " << kernel_W  << ")"<< endl;
      cout << "this kind of padding and kernel_size is not implemented yet." << endl;
      exit(-1);
    }
    int pad_d = padD1;
    int pad_h = padH1;
    int pad_w = padW1;
    int nD = (output_D + 1) / 2;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;
    int kernel_size = (kernel_D + 1 + (kernel_D - 1) / 3) * (kernel_H + 1 + (kernel_H - 1) / 3) * (kernel_W + 1 + (kernel_W - 1) / 3);

    if(kernel_D <= 0 and kernel_H <= 0 and kernel_W <= 0) {
      convLauncherStrideOne3x3(Input.data<T>(), Weight.data<T>(),
                               tmp_input_buffer.data<T>(), tmp_weight_buffer.data<T>(),
                               tmp_product_buffer.data<T>(), tmp_ptr_buffer.data<int64_t>(),
                               B, D, H, W, C, K, kernel_D, kernel_H, kernel_W, pad_d, pad_h, pad_w,
                               Output.data<T>());
    } else if(kernel_D <= 10 and kernel_H <= 10 and kernel_W <= 10) {
      convLauncherStrideOneLarge(Input.data<T>(), Weight.data<T>(),
                               tmp_input_buffer.data<T>(), tmp_weight_buffer.data<T>(),
                               tmp_product_buffer.data<T>(), tmp_ptr_buffer.data<int64_t>(),
                               B, D, H, W, C, K, kernel_D, kernel_H, kernel_W, pad_d, pad_h, pad_w,
                               Output.data<T>());
    } else {
      cout << "kernel_size: " << kernel_H << ", " << kernel_W << endl;
      cout << "stride: 1" << endl;
      cout << "kernel_size fault, not implemented yet." << endl;
      exit(-1);
    }
  } else {
    cout << "stride: " << stride << endl;
    cout << "stride fault, not implemented yet." << endl;
    exit(-1);
  }
}

template void DWM<float>(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer);
template void DWM<at::Half>(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer);
