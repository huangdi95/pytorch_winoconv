#include <cublas_v2.h>
#include <iostream>

#include "src/split_and_concat_launchers.h"
#include "src/split_and_concat_kernels.cu"

using namespace std;

template <>
void winoSplitLauncher<float>(const float* norm_input, int B, int H, int W, int C, int pad_h, int pad_w, int block_H, int block_W, int nH, int nW, float* wino_input) {
    dim3 bDim(C, 1, 1);
    dim3 gDim(nW, nH, B);
//    int size = B*nH*nW* C* block_H* block_W;
//    int size_in = B*H*W*C;
//    float out_host[size];
//    float input_host[size_in];
//    cudaMemcpy(out_host, wino_input, size*sizeof(float), cudaMemcpyDeviceToHost);
//    for(int i = 0; i < size; i++) {
//        cout << out_host[i] << endl;
//    }
//    for(int i = 0; i < size_in; i++) {
//        cout << out_host[i] << endl;
//    }
    winoSplitKernel<<<gDim, bDim>>>(norm_input, B, H, W, C, pad_h, pad_w, block_H, block_W, nH, nW, wino_input);
}

template <>
void winoConcatLauncher<float>(const float* wino_output, int B, int output_H, int output_W, int K, int output_block_H, int output_block_W, float* norm_output) {
    int nH = (output_H+output_block_H-1)/output_block_H;
    int nW = (output_W+output_block_W-1)/output_block_W;
	dim3 bDim(K, 1, 1);
	dim3 gDim(nW, nH, B);
    winoConcatKernel<<<gDim, bDim>>>(wino_output, B, output_H, output_W, K, output_block_H, output_block_W, norm_output);
}
