#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdio.h>
#include <assert.h> 

#include <iostream>
#include <string>

#include "src/base_conv_launchers.h"
#include "src/split_functions.h"
#include "src/split_and_concat_launchers.h"

using namespace std;

template <typename Dtype>
void GWM(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer,
              at::Tensor tmp_feature_buffer, at::Tensor tmp_kernel_buffer, at::Tensor tmp_output_buffer
              ) {
  int B = Input.sizes()[0];
  int H = Input.sizes()[1];
  int W = Input.sizes()[2];
  int C = Input.sizes()[3];
  int K = Weight.sizes()[3];
  int kernel_H = Weight.sizes()[0];
  int kernel_W = Weight.sizes()[1];

  int padH1 = 0;
  int padH2 = 0;
  int padW1 = 0;
  int padW2 = 0;
  
  int output_H = (H + padH1 + padH2 - kernel_H) / stride + 1;
  int output_W = (W + padW1 + padW2 - kernel_W) / stride + 1;

  if(stride == 1) {
    int pad_h = padH1;
    int pad_w = padW1;
	  int nH = (H + 2 * pad_h - kernel_H) / 2 + 1;
    int nW = (W + 2 * pad_w - kernel_W) / 2 + 1;
    int kernel_size = (kernel_H + 1) * (kernel_W + 1);

    if(kernel_H == 1 && kernel_W == 1) {
//    winoSplitLauncher(Input.data<Dtype>(), B, H, W, C, pad_h, pad_w, 4, 4, nH, nW, Output.data<Dtype>());
      convLauncherStrideOne1x1(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 1 && kernel_W == 2) {
      convLauncherStrideOne1x2(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 1 && kernel_W == 3) {
      convLauncherStrideOne1x3(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 2 && kernel_W == 1) {
      convLauncherStrideOne2x1(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 2 && kernel_W == 2) {
      convLauncherStrideOne2x2(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 2 && kernel_W == 3) {
      convLauncherStrideOne2x3(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 3 && kernel_W == 1) {
      convLauncherStrideOne3x1(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 3 && kernel_W == 2) {
      convLauncherStrideOne3x2(Input.data<Dtype>(), Weight.data<Dtype>(), 
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(), 
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 3 && kernel_W == 3) {
      convLauncherStrideOne3x3(Input.data<Dtype>(), Weight.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H, W, K, pad_h, pad_w,
                               Output.data<Dtype>());
    } else if(kernel_H == 5 && kernel_W == 5) {
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 0, H+2*pad_h-2, 0, W+2*pad_w-2, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 0, 3, 0, 3, kernel_H, kernel_W, C, K);
      convLauncherStrideOne3x3(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-2, W+2*pad_w-2, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 0, H+2*pad_h-2, 3, W+2*pad_w, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 0, 3, 3, 5, kernel_H, kernel_W, C, K);
      convLauncherStrideOne3x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-2, W+2*pad_w-3, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 3, H+2*pad_h, 0, W+2*pad_w-2, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 3, 5, 0, 3, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x3(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-3, W+2*pad_w-2, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 3, H+2*pad_h, 3, W+2*pad_w, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 3, 5, 3, 5, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-3, W+2*pad_w-3, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
    } else if(kernel_H == 7 && kernel_W == 7) {
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 0, H+2*pad_h-4, 0, W+2*pad_w-4, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 0, 3, 0, 3, kernel_H, kernel_W, C, K);
      convLauncherStrideOne3x3(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-4, W+2*pad_w-4, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 0, H+2*pad_h-4, 3, W+2*pad_w-2, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 0, 3, 3, 5, kernel_H, kernel_W, C, K);
      convLauncherStrideOne3x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-4, W+2*pad_w-2-3, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 0, H+2*pad_h-4, 5, W+2*pad_w, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 0, 3, 5, 7, kernel_H, kernel_W, C, K);
      convLauncherStrideOne3x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-4, W+2*pad_w-5, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 3, H+2*pad_h-2, 0, W+2*pad_w-4, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 3, 5, 0, 3, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x3(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-2-3, W+2*pad_w-4, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 3, H+2*pad_h-2, 3, W+2*pad_w-2, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 3, 5, 3, 5, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-2-3, W+2*pad_w-2-3, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);

      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 3, H+2*pad_h-2, 5, W+2*pad_w, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 3, 5, 5, 7, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-2-3, W+2*pad_w-5, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 5, H+2*pad_h, 0, W+2*pad_w-4, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 5, 7, 0, 3, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x3(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-5, W+2*pad_w-4, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
  
      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 5, H+2*pad_h, 3, W+2*pad_w-2, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 5, 7, 3, 5, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-5, W+2*pad_w-2-3, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);

      splitFeatureLauncher(Input.data<Dtype>(), tmp_feature_buffer.data<Dtype>(), 5, H+2*pad_h, 5, W+2*pad_w, B, H, W, C, pad_h, pad_w);
      splitFilterLauncher(Weight.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(), 5, 7, 5, 7, kernel_H, kernel_W, C, K);
      convLauncherStrideOne2x2(tmp_feature_buffer.data<Dtype>(), tmp_kernel_buffer.data<Dtype>(),
                               tmp_input_buffer.data<Dtype>(), tmp_weight_buffer.data<Dtype>(),
                               tmp_product_buffer.data<Dtype>(), tmp_ptr_buffer.data<int64_t>(),
                               C, B, H+2*pad_h-5, W+2*pad_w-5, K, 0, 0,
                               tmp_output_buffer.data<Dtype>());
      addFeatureLauncher(Output.data<Dtype>(), Output.data<Dtype>(), tmp_output_buffer.data<Dtype>(), B, output_H, output_W, K);
    }
  }
}

//template void GWM<float>(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
//              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer,
//              at::Tensor tmp_feature_buffer, at::Tensor tmp_kernel_buffer, at::Tensor tmp_output_buffer);
template void GWM<at::Half>(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer,
              at::Tensor tmp_feature_buffer, at::Tensor tmp_kernel_buffer, at::Tensor tmp_output_buffer);
