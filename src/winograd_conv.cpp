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
//time measure
#include <chrono>

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
//    int kernel_size = (kernel_D + 1 + (kernel_D - 1) / 3) * (kernel_H + 1 + (kernel_H - 1) / 3) * (kernel_W + 1 + (kernel_W - 1) / 3);

//    if(kernel_D <= 0 and kernel_H <= 0 and kernel_W <= 0) {
//      convLauncherStrideOne3x3(Input.data<T>(), Weight.data<T>(),
//                               tmp_input_buffer.data<T>(), tmp_weight_buffer.data<T>(),
//                               tmp_product_buffer.data<T>(), tmp_ptr_buffer.data<int64_t>(),
//                               B, D, H, W, C, K, kernel_D, kernel_H, kernel_W, pad_d, pad_h, pad_w,
//                               Output.data<T>());
    int kernel_size1 = int((kernel_D + 1 + (kernel_D - 1) / 3) * (kernel_H + 1 + (kernel_H - 1) / 3) * (kernel_W + 1 + (kernel_W - 1) / 3));
//    auto tmp_input_buffer1 = at::empty({kernel_size1*B*nD*nH*nW*C});
//    auto tmp_weight_buffer1 = at::empty({kernel_size1*C*K});
//    auto tmp_product_buffer1 = at::empty({kernel_size1*nD*nH*nW*B*K});
//    auto tmp_ptr_buffer1 = at::empty({3*kernel_size1});

    if(kernel_D <= 10 and kernel_H <= 10 and kernel_W <= 10) {
      cout << "convLauncheStrideOneLarge!!!!!!" << endl;
      convLauncherStrideOneLarge(Input.data<T>(), Weight.data<T>(),
                               tmp_input_buffer.data<T>(), tmp_weight_buffer.data<T>(),
                               tmp_product_buffer.data<T>(), tmp_ptr_buffer.data<int64_t>(),
                               B, D, H, W, C, K, kernel_D, kernel_H, kernel_W, pad_d, pad_h, pad_w,
                               Output.data<T>());
//    if(kernel_D <= 10 and kernel_H <= 10 and kernel_W <= 10) {
//      convLauncherStrideOneLarge2(Input.data<T>(), Weight.data<T>(),
//                               B, D, H, W, C, K, kernel_D, kernel_H, kernel_W, pad_d, pad_h, pad_w,
//                               Output.data<T>());
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

template <typename T>
void DWM2D(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
         at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer,
         int num_split, at::Tensor H_s, at::Tensor H_e, at::Tensor W_s, at::Tensor W_e, at::Tensor tmp_out_buffer
        ) {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
//  auto tt1 = high_resolution_clock::now();

  int B = Input.sizes()[0];
  int H = Input.sizes()[1];
  int W = Input.sizes()[2];
  int C = Input.sizes()[3];
  int kernel_H = Weight.sizes()[0];
  int kernel_W = Weight.sizes()[1];
  int K = Weight.sizes()[3];

  string padding_ = "VALID" ;
  int padH1 = 0;
  int padH2 = 0;
  int padW1 = 0;
  int padW2 = 0;
  if (padding_ == "SAME") {
    if(stride == 1) {
      padH1 = (kernel_H - 1) / 2;
      padH2 = (kernel_H) / 2;
      padW1 = (kernel_W - 1) / 2;
      padW2 = (kernel_W) / 2;
    } else if(stride == 2) {
      padH1 = (kernel_H - 2 + H%2) / 2;
      padH2 = (kernel_H - 1 + H%2) / 2;
      padW1 = (kernel_W - 2 + W%2) / 2;
      padW2 = (kernel_W - 1 + W%2) / 2;
    }
  } else if (padding_ == "VALID"){
    if(stride == 1) {
      padH1 = 0;
      padH2 = 0;
      padW1 = 0;
      padW2 = 0;
    } else if (stride == 2){
      padH1 = 0;
      padH2 = -(H+kernel_H)%2;
      padW1 = 0;
      padW2 = -(W+kernel_W)%2;
    }
  }
  
  int output_H = (H + padH1 + padH2 - kernel_H) / stride + 1;
  int output_W = (W + padW1 + padW2 - kernel_W) / stride + 1;

  if(stride == 1) {
//  if(kernel_H == 5 && kernel_W == 5) {
//    pad_h = 2;
//    pad_w = 2;
//  }
    if (padH1 != padH2 or padW1 != padW2) {
      cout << "padding == " << padding_ << ", kernel_size == (" << kernel_H << ", " << kernel_W  << ")"<< endl;
      cout << "this kind of padding and kernel_size is not implemented yet." << endl;
      exit(-1);
    }
    int pad_h = padH1;
    int pad_w = padW1;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;
    int kernel_size1 = int((kernel_H + 1 + (kernel_H - 1) / 3) * (kernel_W + 1 + (kernel_W - 1) / 3));
    
//    auto tt2 = high_resolution_clock::now();
//    duration<double, std::milli> ms_double = tt2 - tt1;
//    std::cout << ms_double.count() << " ms" << endl;

    if(kernel_H <= 10 and kernel_W <= 10) {
      convLauncherStrideOneLarge2D2(Input.data<T>(), Weight.data<T>(),
                               tmp_input_buffer.data<T>(), tmp_weight_buffer.data<T>(),
                               tmp_product_buffer.data<T>(), tmp_ptr_buffer.data<int64_t>(),
                               B, H, W, C, K, kernel_H, kernel_W, pad_h, pad_w,
                               Output.data<T>(), num_split, H_s.data<int>(), H_e.data<int>(), W_s.data<int>(), W_e.data<int>(), tmp_out_buffer.data<T>());
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
//template void DWM<at::Half>(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
//              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer);
template void DWM2D<float>(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer, int num_split, at::Tensor H_s, at::Tensor H_e, at::Tensor W_s, at::Tensor W_e, at::Tensor tmp_out_buffer);
//template void DWM2D<at::Half>(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
//              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer);
