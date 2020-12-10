#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "src/add.cuh"
#include "src/utils.hpp"

#include <iostream>

using namespace std;

template <typename Dtype>
void AddGPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c) {
  int N = in_a.numel();
  if (N != in_b.numel())
    throw std::invalid_argument(Formatter()
                                << "Size mismatch A.numel(): " << in_a.numel()
                                << ", B.numel(): " << in_b.numel());


  out_c.resize_({N,1});
  
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

  cout << out_c.sizes() << endl; 
  int n1 = out_c.sizes()[0]; 
  int n2 = out_c.sizes()[1]; 
  cout << n1 << ", " << n2 << endl;

  AddGPUKernel<Dtype>(in_a.data<Dtype>(), in_b.data<Dtype>(),
                      out_c.data<Dtype>(), N, at::cuda::getCurrentCUDAStream());
}

//template void AddGPU<at::Half>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);
template void AddGPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);
