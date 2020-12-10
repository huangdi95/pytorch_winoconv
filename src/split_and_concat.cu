#include <cublas_v2.h>
#include <iostream>

#include "src/split_and_concat_launchers.h"
#include "src/split_and_concat_kernels.cu"

template <>
void winoSplitLauncher<float>(const float* norm_input, int B, int D, int H, int W, int C, int pad_d, int pad_h, int pad_w, int block_D, int block_H, int block_W, int nD, int nH, int nW, float* wino_input) {
    dim3 bDim(C, 1, 1);
    dim3 gDim(nD*nH*nW, B, 1);
    winoSplitKernel<<<gDim, bDim>>>(norm_input, B, D, H, W, C, pad_d, pad_h, pad_w, block_D, block_H, block_W, nD, nH, nW, wino_input);
}

template <>
void winoConcatLauncher<float>(const float* wino_output, int B, int output_D, int output_H, int output_W, int K, int output_block_D, int output_block_H, int output_block_W, float* norm_output) {
    int nD = (output_D+output_block_D-1)/output_block_D;
    int nH = (output_H+output_block_H-1)/output_block_H;
    int nW = (output_W+output_block_W-1)/output_block_W;
	dim3 bDim(K, 1, 1);
	dim3 gDim(nD*nH*nW, B, 1);
    winoConcatKernel<<<gDim, bDim>>>(wino_output, B, output_D, output_H, output_W, K, output_block_D, output_block_H, output_block_W, norm_output);
}

template <>
void winoSplitLauncher2d<float>(const float* norm_input, int B, int H, int W, int C, int pad_h, int pad_w, int block_H, int block_W, int nH, int nW, float* wino_input) {
    dim3 bDim(C, 1, 1);
    dim3 gDim(nH*nW, B, 1);
    winoSplitKernel2d<<<gDim, bDim>>>(norm_input, B, H, W, C, pad_h, pad_w, block_H, block_W, nH, nW, wino_input);
}

template <>
void winoConcatLauncher2d<float>(const float* wino_output, int B, int output_H, int output_W, int K, int output_block_H, int output_block_W, float* norm_output) {
    int nH = (output_H+output_block_H-1)/output_block_H;
    int nW = (output_W+output_block_W-1)/output_block_W;
	dim3 bDim(K, 1, 1);
	dim3 gDim(nH*nW, B, 1);
    winoConcatKernel2d<<<gDim, bDim>>>(wino_output, B, output_H, output_W, K, output_block_H, output_block_W, norm_output);
}
