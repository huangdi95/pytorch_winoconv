#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include "conv_base.cu"
#include "fused_kernels.cu"
#include "composer_kernels.cu"
//time measure
#include <chrono>
#define CHECK_RESULT 0
#define MY 1
#define KERNEL 0
#define SHIFT 0
#define BN16 0
#define SPLIT 0
#define ADD 0
//#define BN 32
//#define BC 8
//#define BK 64
//#define BN 32*32
//#define BC 8*32
#define Bi 32    //input batch
#define Hi 112  //input h
#define Wi 256 //input w
#define BC 7*3 //input c
#define BK 64   //output c
#define PH 1    //pad h
#define PW 1    //pad w
#define FILTER 9 //filter size

void randomInit(float*, int);
void randomInit1(float*, int);
void printDiff(float*, float*, int, int, int, int);

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
__global__ void winograd2DFused_kernel(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split, int size_C) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    float accu[16/8][bk] = {0};

    extern __shared__ float smem[]; // [16, bn/2, bk/2+1] with 0.5 conflict padding, the last dimension is bk/4

    float *input_smem = smem;
    float *output_smem = smem;
    float *agg_smem = smem + 16*16*32;

    int yBase = 2 * (by / nW) - pad_h;
    int xBase = 2 * (by % nW) - pad_w;
    int f_x, f_y;
    float prefetch1[16];
    float *input_patch = &prefetch1[0];
/////////////// prefetch /////////////
    for(int j = 0; j < (splitH + 1); j++) {
      for(int k = 0; k < (splitW + 1); k++) {
        f_y = yBase + j;
        f_x = xBase + k;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
          prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
//          prefetch1[j * (splitW+1) + k] = float(0);
        } else {
          prefetch1[j * (splitW+1) + k] = float(0);
        }
      }
    }

    /****************************************************/
    /*  gmem                                            */
    /****************************************************/
    kernel_16_gmem<bn, bc, bk, 3, 3, 0, 3>(input, input_patch, input_smem, weight, output_smem, output, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
    kernel_16_gmem<bn, bc, bk, 3, 3, 3, 6>(input, input_patch, input_smem, weight, output_smem, output+size_C, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
    kernel_16_gmem<bn, bc, bk, 3, 3, 6, 8>(input, input_patch, input_smem, weight, output_smem, output+2*size_C, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
    kernel_16_gmem<bn, bc, bk, 3, 3, 8, 9>(input, input_patch, input_smem, weight, output_smem, output+3*size_C, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
    /****************************************************/
    /*  smem                                            */
    /****************************************************/
//    kernel_16_smem<bn, bc, bk, 3, 3, 0, 3>(input, input_patch, input_smem, weight, output_smem, agg_smem, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
//    kernel_16_smem<bn, bc, bk, 3, 3, 3, 6>(input, input_patch, input_smem, weight, output_smem, agg_smem, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
//    kernel_16_smem<bn, bc, bk, 3, 3, 6, 8>(input, input_patch, input_smem, weight, output_smem, agg_smem, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
//    kernel_16_smem<bn, bc, bk, 3, 3, 8, 9>(input, input_patch, input_smem, weight, output_smem, agg_smem, H_start, W_start, kernel_stride, xBase, yBase, nH, nW, B, H, W, C, K, output_H, output_W, bx, by, warp_id, lane_id, num_split, H_end, W_end);
//    storeToGlobal_bn16<bn, bc, bk, splitH, splitW>(agg_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id);
}

void winograd2DFused_split(const float *d_A, const float *tmp_weight_buffer_fused, float *d_C, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int NH, int NW, int B, int H, int W, int C, int K, int Ho, int Wo, int pad_h, int pad_w, int num_split, cudaStream_t stream1, cudaStream_t stream2, cudaStream_t stream3, cudaStream_t stream4, int size_C) {
    const int BN=NH*NW*Bi;  //N 
    const int bn = 32;
    const int bc = 8;
    const int bk = 64;
    const int maxbytes = 67584;//16 * (bn * bc + bn * 33) * 4; // 82 KB
//    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 0, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 3, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 6, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 8, 9>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    winograd2DFused_16<bn, bc, bk, 3, 3, 0, 3><<<(BN/bn)*(BK/bk), 256, maxbytes, stream1>>>(d_A, tmp_weight_buffer_fused, d_C, kernel_stride, H_start, H_end, W_start, W_end, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split);
    winograd2DFused_16<bn, bc, bk, 3, 3, 3, 6><<<(BN/bn)*(BK/bk), 256, maxbytes, stream2>>>(d_A, tmp_weight_buffer_fused, d_C+size_C, kernel_stride, H_start, H_end, W_start, W_end, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split);
    winograd2DFused_16<bn, bc, bk, 3, 3, 6, 8><<<(BN/bn)*(BK/bk), 256, maxbytes, stream3>>>(d_A, tmp_weight_buffer_fused, d_C+2*size_C, kernel_stride, H_start, H_end, W_start, W_end, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split);
    winograd2DFused_16<bn, bc, bk, 3, 3, 8, 9><<<(BN/bn)*(BK/bk), 256, maxbytes, stream4>>>(d_A, tmp_weight_buffer_fused, d_C+3*size_C, kernel_stride, H_start, H_end, W_start, W_end, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split);
    


}
//template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
//__global__ void winograd2DFused(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
//    int warp_id = threadIdx.x / 32;
//    int lane_id = threadIdx.x % 32;
//    int bx = blockIdx.x % (K / bk);
//    int by = blockIdx.x / (K / bk);  //TODO: slow???
//
//    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding
//
//    float *input_smem = smem;
//    float *output_smem = smem + 16 * bc * bn;
//
//    int yBase = 2 * (by / nW) - pad_h;
//    int xBase = 2 * (by % nW) - pad_w;
//    int f_x, f_y;
//    float prefetch1[16];
//    float *input_patch = &prefetch1[0];
///////////////// prefetch /////////////
//    for(int j = 0; j < (splitH + 1); j++) {
//      for(int k = 0; k < (splitW + 1); k++) {
//        f_y = yBase + j;
//        f_x = xBase + k;
//        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
//          prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
//        } else {
//          prefetch1[j * (splitW+1) + k] = float(0);
//        }
//      }
//    }
///////////////////////////////////////
//    int count = 0;
//    kernel3x3<bn, bc, bk, 9>(input_patch, prefetch1, input_smem, weight, output_smem, output, H_start, W_start, xBase, yBase, nH, nW, B, C, output_H, output_W, bx, by, warp_id, lane_id, num_split, count); 
//}

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
__global__ void winograd2DFused_shift(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    float accu[16/8][bk] = {0};

    extern __shared__ float smem[]; // [16, bn/2, bk/2+1] with 0.5 conflict padding, the last dimension is bk/4

    float *input_smem = smem;
    float *output_smem = smem;
    float *agg_smem = smem + 16*32*32;
//    float agg_smem[4*64*32];

    int yBase = 2 * (by / nW) - pad_h;
    int xBase = 2 * (by % nW) - pad_w;
    int f_x, f_y;
    float prefetch1[16];
    float *input_patch = &prefetch1[0];
/////////////// prefetch /////////////
    for(int j = 0; j < (splitH + 1); j++) {
      for(int k = 0; k < (splitW + 1); k++) {
        f_y = yBase + j;
        f_x = xBase + k;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
          prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
        } else {
          prefetch1[j * (splitW+1) + k] = float(0);
        }
      }
    }
/////////////////////////////////////
    for (int count = 0; count < C * num_split; count+=bc) {
        int bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<bn, bc, splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * num_split) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
          for(int m = 0; m < (splitH + 1); m++) {
            for(int n = 0; n < (splitW + 1); n++) {
              f_y = yBase + m + H_start[bz2];
              f_x = xBase + n + W_start[bz2];
              if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
                prefetch1[m * (splitW + 1) + n] = input[((((warp_id + i2) * H + f_y) * W + f_x)) * B + lane_id];
//                prefetch1[m * (splitW + 1) + n] = float(0);
              } else {
                prefetch1[m * (splitW + 1) + n] = float(0);
              }
            }
          }
        }
/////////////////////////////////////

        float *ip = &input_smem[2 * warp_id * bc * bn];
        const float *wp = &weight[(2 * warp_id + kernel_stride[bz]) * C * K + lane_id + i * K + bx * bk];
///////////// batched matmul bcbn 32x2x8 outer product//////////////
#pragma unroll
        for(int k = 0; k < 16/8; k++) {
#pragma unroll
          for(int j = 0; j < bc; j++) {
              float wv = wp[0];
              for(int l = 0; l < 32; l++) {
                accu[k][l] += ip[l] * wv;
              }
              if (bk > 32) {
                wp += 32; 
                wv = wp[0];
                for(int l = 32; l < 64; l++) {
                  accu[k][l] += ip[l-32] * wv;
                }
                wp += (K - 32);
              } else if (bk <= 32) {
                wp += K;
              }
              ip += bn;
          }
          wp += (C - bc) * K;
        }
        __syncthreads();
////////////////////////////////////////////////////////////////
    }
    for (int i = 0; i < bk; i += 32) {
        //////// load wino output /////
        unsigned int offset = by * bn * K + bx * bk;
        for (int j = 0; j < bn; j++) {
            output_smem[((2 * warp_id) * bn + j) * (32) + (lane_id+j)%32] = accu[0][j + i];
            output_smem[((2 * warp_id + 1) * bn + j) * (32) + (lane_id+j)%32] = accu[1][j + i];
        }
//    for (int i = 0; i < bk; i += 16) {
//        //////// load wino output /////
//        unsigned int offset = by * bn * K + bx * bk;
//        for (int j = 0; j < bn/2; j++) {
//            if (i%32 <= lane_id  && lane_id <= (i + 15) % 32) {
//                output_smem[((2 * warp_id) * bn + j) * (32 + 1) + lane_id] = accu[0][2 * j + int(i/32)*32];
//                output_smem[((2 * warp_id) * bn + j) * (32 + 1) + 16 + lane_id] = accu[0][2 * j + 1 + int(i/32)*32];
//                output_smem[((2 * warp_id + 1) * bn + j) * (32 + 1) + lane_id] = accu[1][2 * j + int(i/32)*32];
//                output_smem[((2 * warp_id + 1) * bn + j) * (32 + 1) + 16 + lane_id] = accu[1][2 * j + 1 + int(i/32)*32];
//            }
//        }
        __syncthreads();
        //////// output transform //////
//        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
        outputWino2NormTransform2D_fused2_shift<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i, 0);
        __syncthreads();
    }
    storeToGlobal_shift<bn, bc, bk, splitH, splitW>(agg_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id);
}

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
__global__ void winograd2DFused_bn16(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
//    float accu[16/8][bk] = {0};

    extern __shared__ float smem[]; // [16, bn/2, bk/2+1] with 0.5 conflict padding, the last dimension is bk/4

    float *input_smem = smem;
    float *output_smem = smem;
    float *agg_smem = smem + 16*16*32;

    int yBase = 2 * (by / nW) - pad_h;
    int xBase = 2 * (by % nW) - pad_w;
    int f_x, f_y;
    float prefetch1[16];
    float *input_patch = &prefetch1[0];
/////////////// prefetch /////////////
    for(int j = 0; j < (splitH + 1); j++) {
      for(int k = 0; k < (splitW + 1); k++) {
        f_y = yBase + j;
        f_x = xBase + k;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
          prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
//          prefetch1[j * (splitW+1) + k] = float(0);
        } else {
          prefetch1[j * (splitW+1) + k] = float(0);
        }
      }
    }
/////////////////////////////////////
    for (int split = 0; split < 9; split++) {
    float accu[16/8][bk] = {0};
    for (int count = C*split; count < C * (split+1); count+=bc) {
        int bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<bn, bc, splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * num_split) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
          for(int m = 0; m < (splitH + 1); m++) {
            for(int n = 0; n < (splitW + 1); n++) {
              f_y = yBase + m + H_start[bz2];
              f_x = xBase + n + W_start[bz2];
              if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
                prefetch1[m * (splitW + 1) + n] = input[((((warp_id + i2) * H + f_y) * W + f_x)) * B + lane_id];
//                prefetch1[m * (splitW + 1) + n] = float(0);
              } else {
                prefetch1[m * (splitW + 1) + n] = float(0);
              }
            }
          }
        }
/////////////////////////////////////

        float *ip = &input_smem[2 * warp_id * bc * bn];
        const float *wp = &weight[(2 * warp_id + (splitH + 1) * (splitW + 1) * bz) * C * K + lane_id + i * K + bx * bk];
///////////// batched matmul bcbn 32x2x8 outer product//////////////
#pragma unroll
        for(int k = 0; k < 16/8; k++) {
#pragma unroll
          for(int j = 0; j < bc; j++) {
              float wv = wp[0];
              for(int l = 0; l < 32; l++) {
                accu[k][l] += ip[l] * wv;
              }
              if (bk > 32) {
                wp += 32; 
                wv = wp[0];
                for(int l = 32; l < 64; l++) {
                  accu[k][l] += ip[l-32] * wv;
                }
                wp += (K - 32);
              } else if (bk <= 32) {
                wp += K;
              }
              ip += bn;
          }
          wp += (C - bc) * K;
        }
        __syncthreads();
////////////////////////////////////////////////////////////////
    }
    for (int i = 0; i < bk; i += 32) {
        //////// load wino output /////
        unsigned int offset = by * bn * K + bx * bk;
        for (int j = 0; j < bn/2; j++) {
            output_smem[((2 * warp_id) * bn/2 + j) * (32) + lane_id] = accu[0][j + i];
            output_smem[((2 * warp_id + 1) * bn/2 + j) * (32) + lane_id] = accu[1][j + i];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused2_bn16<bn, bc, bk, splitH, splitW>(output_smem, agg_smem, kernel_stride, H_start, W_start, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i, 0);
        __syncthreads();
        //////////////////////////////
        for (int j = bn/2; j < bn; j++) {
            output_smem[((2 * warp_id) * bn/2 + j-bn/2) * (32) + lane_id] = accu[0][j + i];
            output_smem[((2 * warp_id + 1) * bn/2 + j-bn/2) * (32) + lane_id] = accu[1][j + i];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused2_bn16<bn, bc, bk, splitH, splitW>(output_smem, agg_smem, kernel_stride, H_start, W_start, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i, 16);
        __syncthreads();
        //////////////////////////////
    }
    }
    storeToGlobal_bn16<bn, bc, bk, splitH, splitW>(agg_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id);
}
// double output trans test
//template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
//__global__ void winograd2DFused(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
//    int warp_id = threadIdx.x / 32;
//    int lane_id = threadIdx.x % 32;
//    int bx = blockIdx.x % (K / bk);
//    int by = blockIdx.x / (K / bk);  //TODO: slow???
//    float accu[16/8][bk] = {0};
//
//    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding
//
//    float *input_smem = smem;
//    float *output_smem = smem;
//
//    int yBase = 2 * (by / nW) - pad_h;
//    int xBase = 2 * (by % nW) - pad_w;
//    int f_x, f_y;
//    float prefetch1[16];
//    float *input_patch = &prefetch1[0];
///////////////// prefetch /////////////
//    for(int j = 0; j < (splitH + 1); j++) {
//      for(int k = 0; k < (splitW + 1); k++) {
//        f_y = yBase + j;
//        f_x = xBase + k;
//        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
//          prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
//        } else {
//          prefetch1[j * (splitW+1) + k] = float(0);
//        }
//      }
//    }
///////////////////////////////////////
//    for (int count = 0; count < C * num_split; count+=bc) {
//        int bz = count / C;
//        int i = count % C;
//   
//        //////// input transform //////
//        inputNorm2WinoTransform2D_fused<bn, bc, splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
//        __syncthreads();
//        //////////////////////////////
//
///////////////// prefetch /////////////
//        if (count+bc < C * num_split) {
//          int bz2 = (count + bc) / C;
//          int i2 = (count + bc) % C;
//          for(int m = 0; m < (splitH + 1); m++) {
//            for(int n = 0; n < (splitW + 1); n++) {
//              f_y = yBase + m + H_start[bz2];
//              f_x = xBase + n + W_start[bz2];
//              if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
//                prefetch1[m * (splitW + 1) + n] = input[((((warp_id + i2) * H + f_y) * W + f_x)) * B + lane_id];
////                prefetch1[m * (splitW + 1) + n] = float(0);
//              } else {
//                prefetch1[m * (splitW + 1) + n] = float(0);
//              }
//            }
//          }
//        }
///////////////////////////////////////
//
//        float *ip = &input_smem[2 * warp_id * bc * bn];
//        const float *wp = &weight[(2 * warp_id + (splitH + 1) * (splitW + 1) * bz) * C * K + lane_id + i * K + bx * bk];
/////////////// batched matmul bcbn 32x2x8 outer product//////////////
//#pragma unroll
//        for(int k = 0; k < 16/8; k++) {
//#pragma unroll
//          for(int j = 0; j < bc; j++) {
//              float wv = wp[0];
//              for(int l = 0; l < 32; l++) {
//                accu[k][l] += ip[l] * wv;
//              }
//              if (bk > 32) {
//                wp += 32; 
//                wv = wp[0];
//                for(int l = 32; l < 64; l++) {
//                  accu[k][l] += ip[l-32] * wv;
//                }
//                wp += (K - 32);
//              } else if (bk <= 32) {
//                wp += K;
//              }
//              ip += bn;
//          }
//          wp += (C - bc) * K;
//        }
//        __syncthreads();
//////////////////////////////////////////////////////////////////
//    }
//    for (int i = 0; i < bk; i += 32) {
//        //////// load wino output /////
//        unsigned int offset = by * bn * K + bx * bk;
//        for (int j = 0; j < bn; j++) {
//            output_smem[((2 * warp_id) * bn + j) * (32) + (lane_id+j)%32] = accu[0][j + i];
//            output_smem[((2 * warp_id + 1) * bn + j) * (32) + (lane_id+j)%32] = accu[1][j + i];
//        }
//        __syncthreads();
//        //////// output transform //////
//        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
//        __syncthreads();
//        //////////////////////////////
//    }
//}

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
__global__ void winograd2DFused(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    float accu[16/8][bk] = {0};

    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding

    float *input_smem = smem;
    float *output_smem = smem;

    int yBase = 2 * (by / nW) - pad_h;
    int xBase = 2 * (by % nW) - pad_w;
    int f_x, f_y;
    float prefetch1[16];
    float *input_patch = &prefetch1[0];
/////////////// prefetch /////////////
    if (bc == 8) {
      for(int j = 0; j < (splitH + 1); j++) {
        for(int k = 0; k < (splitW + 1); k++) {
          f_y = yBase + j;
          f_x = xBase + k;
          if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
          } else {
            prefetch1[j * (splitW+1) + k] = float(0);
          }
        }
      }
    } else {
      if (warp_id < bc){
        for(int j = 0; j < (splitH + 1); j++) {
          for(int k = 0; k < (splitW + 1); k++) {
            f_y = yBase + j;
            f_x = xBase + k;
            if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
              prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
            } else {
              prefetch1[j * (splitW+1) + k] = float(0);
            }
          }
        }
      }
    }
/////////////////////////////////////
    for (int count = 0; count < C * num_split; count+=bc) {
        int bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        if (bc == 8) {
          inputNorm2WinoTransform2D_fused<bn, bc, splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        } else {
          if (warp_id < bc){
            inputNorm2WinoTransform2D_fused<bn, bc, splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
          }
        }
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * num_split) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
          if (bc == 8) {
            for(int m = 0; m < (splitH + 1); m++) {
              for(int n = 0; n < (splitW + 1); n++) {
                f_y = yBase + m + H_start[bz2];
                f_x = xBase + n + W_start[bz2];
                if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
                  prefetch1[m * (splitW + 1) + n] = input[((((warp_id + i2) * H + f_y) * W + f_x)) * B + lane_id];
                } else {
                  prefetch1[m * (splitW + 1) + n] = float(0);
                }
              }
            }
          } else {
            if (warp_id < bc){
              for(int m = 0; m < (splitH + 1); m++) {
                for(int n = 0; n < (splitW + 1); n++) {
                  f_y = yBase + m + H_start[bz2];
                  f_x = xBase + n + W_start[bz2];
                  if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
                    prefetch1[m * (splitW + 1) + n] = input[((((warp_id + i2) * H + f_y) * W + f_x)) * B + lane_id];
                  } else {
                    prefetch1[m * (splitW + 1) + n] = float(0);
                  }
                }
              }
            }
          }
        }
/////////////////////////////////////

        float *ip = &input_smem[2 * warp_id * bc * bn];
        const float *wp = &weight[(2 * warp_id + (splitH + 1) * (splitW + 1) * bz) * C * K + lane_id + i * K + bx * bk];
///////////// batched matmul bcbn 32x2x8 outer product//////////////
#pragma unroll
        for(int k = 0; k < 16/8; k++) {
#pragma unroll
          for(int j = 0; j < bc; j++) {
              float wv = wp[0];
              for(int l = 0; l < 32; l++) {
                accu[k][l] += ip[l] * wv;
              }
              if (bk > 32) {
                wp += 32; 
                wv = wp[0];
                for(int l = 32; l < 64; l++) {
                  accu[k][l] += ip[l-32] * wv;
                }
                wp += (K - 32);
              } else if (bk <= 32) {
                wp += K;
              }
              ip += bn;
          }
          wp += (C - bc) * K;
        }
        __syncthreads();
////////////////////////////////////////////////////////////////
    }
    for (int i = 0; i < bk; i += 32) {
        //////// load wino output /////
        unsigned int offset = by * bn * K + bx * bk;
        for (int j = 0; j < bn; j++) {
            output_smem[((2 * warp_id) * bn + j) * (32 + 1) + lane_id] = accu[0][j + i];
            output_smem[((2 * warp_id + 1) * bn + j) * (32 + 1) + lane_id] = accu[1][j + i];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////
    }
}

int main() {
    /****************************************************/
    /*  Preparations                                    */
    /****************************************************/
    
    int Ho=Hi-FILTER+1+2*PH;//TODO
    int Wo=Wi-FILTER+1+2*PW;
    int NH=(Ho+1)/2; //nH
    int NW=(Wo+1)/2; //nW
    int BN=NH*NW*Bi;  //N 

    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    // set seed for rand()
    srand(200);

    // allocate host memory for matrices A and B
    unsigned int size_A = Bi * Hi * Wi * BC;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = FILTER * FILTER * BC * BK;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // allocate device memory for result
    unsigned int size_C = BK * Ho * Wo * Bi;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);
    for (int i = 0; i < size_C; i++) {
        h_C[i] = 0;
    }
    cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
#if FILTER == 3
    int H_s[] = {0}; 
    int H_e[] = {3}; 
    int W_s[] = {0}; 
    int W_e[] = {3}; 
#elif FILTER == 6
    int H_s[] = {0, 0, 3, 3};
    int H_e[] = {3, 3, 6, 6};
    int W_s[] = {0, 3, 0, 3};
    int W_e[] = {3, 6, 3, 6};
#elif FILTER == 7
    int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6};
    int H_e[] = {3, 3, 3, 6, 6, 6, 7, 7, 7};
    int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6};
    int W_e[] = {3, 6, 7, 3, 6, 7, 3, 6, 7};
#elif FILTER == 9
    int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6};
    int H_e[] = {3, 3, 3, 6, 6, 6, 9, 9, 9};
    int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6};
    int W_e[] = {3, 6, 9, 3, 6, 9, 3, 6, 9};
#endif
    const int num_split = sizeof(W_s) / sizeof(W_s[0]);
    const int N = num_split * BN * BK;
    cout << num_split << endl;

    int kernel_stride[num_split] = {0};
    for(int i = 1; i < num_split; i++) {
        kernel_stride[i] = (H_e[i-1] - H_s[i-1] + 1) * (W_e[i-1] - W_s[i-1] + 1) + kernel_stride[i-1];
    }
    int Batch = kernel_stride[num_split-1] + (H_e[num_split-1] - H_s[num_split-1] + 1) * (W_e[num_split-1] - W_s[num_split-1] + 1);

    float flop = 2 * (float)Bi * (float)NH * (float)NW *(float)BC * (float)BK * Batch;
    printf("wino flop: %f\n", flop);

    int *H_start_gpu = nullptr;
    int *W_start_gpu = nullptr;
    int *H_end_gpu = nullptr;
    int *W_end_gpu = nullptr;
    int *kernel_stride_gpu = nullptr;
    cudaMalloc((void**)&H_start_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&H_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&W_start_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&W_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&kernel_stride_gpu, num_split*sizeof(int));
    cudaMemcpy(H_start_gpu, H_s, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(W_start_gpu, W_s, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(H_end_gpu, H_e, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(W_end_gpu, W_e, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_stride_gpu, kernel_stride, num_split*sizeof(int), cudaMemcpyHostToDevice);

#if MY == 1
    /****************************************************/
    /*  My kernel                                       */
    /****************************************************/

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C,
                              cudaMemcpyHostToDevice);


    float* tmp_weight_buffer_fused;
    cudaMalloc((void**) &tmp_weight_buffer_fused, Batch*BC*BK*sizeof(float));
    cudaDeviceSynchronize();
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    // naive implementation
    dim3 bDim2(BK, 1, 1);
    dim3 gDim2(BC, num_split, 1);
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, FILTER, FILTER, BC, BK);

    const int bn = 32;
    const int bc = 7;
    const int bk = 64;
    const int maxbytes = 67584;//16 * (bn * bc + bn * 33) * 4; // 82 KB
    cudaFuncSetAttribute(winograd2DFused<bn, bc, bk, 3, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    winograd2DFused<bn, bc, bk, 3, 3><<<(BN/bn)*(BK/bk), 256, maxbytes>>>(d_A, tmp_weight_buffer_fused, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split);
    
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("My\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++) {
        cout << h_C[i] << " "; 
    }
    cout << endl;
    cudaDeviceSynchronize();
#endif

#if SHIFT == 1
    /****************************************************/
    /*  My kernel                                       */
    /****************************************************/

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C,
                              cudaMemcpyHostToDevice);


    float* tmp_weight_buffer_fused;
    cudaMalloc((void**) &tmp_weight_buffer_fused, Batch*BC*BK*sizeof(float));
    cudaDeviceSynchronize();
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    // naive implementation
    dim3 bDim2(BK, 1, 1);
    dim3 gDim2(BC, num_split, 1);
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, FILTER, FILTER, BC, BK);

    const int bn = 32;
    const int bc = 8;
    const int bk = 64;
    const int maxbytes = 98304;//16 * (bn * bc + bn * 33) * 4; // 82 KB
    cudaFuncSetAttribute(winograd2DFused_shift<bn, bc, bk, 3, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    winograd2DFused_shift<bn, bc, bk, 3, 3><<<(BN/bn)*(BK/bk), 256, maxbytes>>>(d_A, tmp_weight_buffer_fused, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split);
    
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("SHIFT\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++) {
        cout << h_C[i] << " "; 
    }
    cout << endl;
    cudaDeviceSynchronize();
#endif

#if BN16 == 1
    /****************************************************/
    /*  My kernel                                       */
    /****************************************************/

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C,
                              cudaMemcpyHostToDevice);


    #if SHIFT == 0
    float* tmp_weight_buffer_fused;
    cudaMalloc((void**) &tmp_weight_buffer_fused, Batch*BC*BK*sizeof(float));
    dim3 bDim2(BK, 1, 1);
    dim3 gDim2(BC, num_split, 1);
    const int bn = 32;
    const int bc = 8;
    const int bk = 64;
    const int maxbytes = 66048;//16 * (bn * bc + bn * 33) * 4; // 82 KB
    #endif
    cudaDeviceSynchronize();
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    // naive implementation
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, FILTER, FILTER, BC, BK);

    cudaFuncSetAttribute(winograd2DFused_bn16<bn, bc, bk, 3, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    winograd2DFused_bn16<bn, bc, bk, 3, 3><<<(BN/bn)*(BK/bk), 256, maxbytes>>>(d_A, tmp_weight_buffer_fused, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split);
    
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("BN16\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++) {
        cout << h_C[i] << " "; 
    }
    cout << endl;
    cudaDeviceSynchronize();
#endif

#if KERNEL == 1
    /****************************************************/
    /*  My kernel                                       */
    /****************************************************/

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C,
                              cudaMemcpyHostToDevice);


    float* tmp_output_buffer;
    cudaMalloc((void**) &tmp_output_buffer, 4*size_C*sizeof(float));
    #if SHIFT == 0 and BN16 == 0
    float* tmp_weight_buffer_fused;
    cudaMalloc((void**) &tmp_weight_buffer_fused, Batch*BC*BK*sizeof(float));
    dim3 bDim2(BK, 1, 1);
    dim3 gDim2(BC, num_split, 1);
    const int bn = 32;
    const int bc = 8;
    const int bk = 64;
    const int maxbytes = 66048;//16 * (bn * bc + bn * 33) * 4; // 82 KB
    #endif
    // warmup
    cudaFuncSetAttribute(winograd2DFused_kernel<bn, bc, bk, 3, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    for (int i = 0; i < 50; i++) {
        wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, FILTER, FILTER, BC, BK);
        winograd2DFused_kernel<bn, bc, bk, 3, 3><<<(BN/bn)*(BK/bk), 256, maxbytes>>>(d_A, tmp_weight_buffer_fused, tmp_output_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split, size_C);
        agg <<<(size_C + 256 - 1) / 256, 256>>> (tmp_output_buffer, d_C, size_C);
    }
    for (int i = 0; i < size_C; i++) {
        h_C[i] = 0;
    }
    cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    for (int i = 0; i < 100; i++) {
        wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, FILTER, FILTER, BC, BK);

        winograd2DFused_kernel<bn, bc, bk, 3, 3><<<(BN/bn)*(BK/bk), 256, maxbytes>>>(d_A, tmp_weight_buffer_fused, tmp_output_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split, size_C);
        agg <<<(size_C + 256 - 1) / 256, 256>>> (tmp_output_buffer, d_C, size_C);
    }
    
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("BN16\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / (msecTotal/100)/ 1e+6);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++) {
        cout << h_C[i] << " "; 
    }
    cout << endl;
    cudaDeviceSynchronize();
#endif

#if SPLIT == 1
    /****************************************************/
    /*  My kernel                                       */
    /****************************************************/

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C,
                              cudaMemcpyHostToDevice);


    float* tmp_weight_buffer_fused;
    cudaMalloc((void**) &tmp_weight_buffer_fused, Batch*BC*BK*sizeof(float));
    float* tmp_output_buffer;
    cudaMalloc((void**) &tmp_output_buffer, 4*size_C*sizeof(float));
    const int bn = 32;
    const int bc = 8;
    const int bk = 64;

    const int maxbytes = 67584;//16 * (bn * bc + bn * 33) * 4; // 82 KB
    cudaStream_t stream1, stream2, stream3, stream4 ;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    // warmup
    dim3 bDim2(BK, 1, 1);
    dim3 gDim2(BC, num_split, 1);
    for (int i = 0; i < 50; i++) {
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 0, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 3, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 6, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 8, 9>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, FILTER, FILTER, BC, BK);
        winograd2DFused_split(d_A, tmp_weight_buffer_fused, tmp_output_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split, stream1, stream2, stream3, stream4, size_C);
        agg <<<(size_C + 256 - 1) / 256, 256>>> (tmp_output_buffer, d_C, size_C);
    }
    // create and start timer
    for (int i = 0; i < size_C; i++) {
        h_C[i] = 0;
    }
    cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    for (int i = 0; i < 100; i++) {
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 0, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 3, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 6, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    cudaFuncSetAttribute(winograd2DFused_16<bn, bc, bk, 3, 3, 8, 9>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//        cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
        wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, FILTER, FILTER, BC, BK);
        winograd2DFused_split(d_A, tmp_weight_buffer_fused, tmp_output_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW, num_split, stream1, stream2, stream3, stream4, size_C);
        agg <<<(size_C + 256 - 1) / 256, 256>>> (tmp_output_buffer, d_C, size_C);
    }
    
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("My\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / (msecTotal / 100) / 1e+6);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++) {
        cout << h_C[i] << " "; 
    }
    cout << endl;
    cudaDeviceSynchronize();
#endif

#if ADD == 1
    size_C = BK * Ho * Wo * Bi;

    float* tmp_buffer;
    cudaMalloc((void**) &tmp_buffer, 4*size_C*sizeof(float));
    float* tmp_buffer2;
    cudaMalloc((void**) &tmp_buffer2, size_C*sizeof(float));
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    // naive implementation
    cout << size_C << endl;
    agg <<<(size_C + 256 - 1) / 256, 256>>> (tmp_buffer, tmp_buffer2, size_C);

    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("My add\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, float(size_C)*5 / msecTotal/ 1e+6);
#endif

#if CHECK_RESULT == 1
    /****************************************************/
    /*  Baseline                                        */
    /****************************************************/

    float* tmp_input_buffer;
    float* tmp_weight_buffer;
    float* tmp_product_buffer;
    float* tmp_out_buffer;
    int64_t* tmp_ptr_buffer;
    cudaMalloc((void**) &tmp_input_buffer, Batch*Bi*NH*NW*BC*sizeof(float));
    cudaMalloc((void**) &tmp_weight_buffer, Batch*BC*BK*sizeof(float));
    cudaMalloc((void**) &tmp_product_buffer, Batch*Bi*NH*NW*BK*sizeof(float));
    cudaMalloc((void**) &tmp_out_buffer, num_split*Bi*Ho*Wo*BK*sizeof(float));
    cudaMalloc((void**) &tmp_ptr_buffer, 3*Batch*sizeof(int64_t));

    cublasHandle_t handle;
    cublasCreate(&handle);
    // copy host memory to device
    
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
////////////TODO: transform ///////////////
    convLauncherStrideOneLarge2D_base<float> (d_A, d_B, tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, tmp_ptr_buffer, Bi, Hi, Wi, BC, BK, FILTER, FILTER, PH, PW, d_C, num_split, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, tmp_out_buffer, handle);
///////////////////////////////////////////
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Baseline\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
    float* ref = (float*) malloc(mem_size_C);
    cudaMemcpy(ref, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++) {
        cout << ref[i] << " "; 
    }
    cout << endl;
    cudaDeviceSynchronize();
#endif

    // check result
#if CHECK_RESULT == 1
#if FAST_BASELINE == 0
    printDiff(ref, h_C, BK, Ho, Wo, Bi);
#endif
    free(ref);
    cudaFree(tmp_input_buffer);
    cudaFree(tmp_weight_buffer);
    cudaFree(tmp_product_buffer);
    cudaFree(tmp_out_buffer);
    cudaFree(tmp_ptr_buffer);
#endif
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
#if MY == 1
    cudaFree(tmp_weight_buffer_fused);
#endif
    return 0;
}

void printDiff(float *data1, float *data2, int d0, int d1, int d2, int d3)
{
  int i,j,idx,l,x;
  int error_count=0;
    
  for (l=0; l<d0; l++) {
    for (j=0; j<d1; j++) {
      for (i=0; i<d2; i++) {
        for (x=0; x<d3; x++) {
          idx = l*d1*d2*d3+j*d2*d3+i*d3+x;
          if (fabs(data1[idx] - data2[idx]) > 0.001 ) {
             printf("diff(%d,%d,%d,%d) CPU=%4.4f, GPU=%4.4f \n", l,j,i,x, data1[idx], data2[idx]);
             error_count++;
          }
        }
      }
    }
  }
  printf("Total Errors = %d \n", error_count);
}

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void randomInit1(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = 1;//rand() / (float)RAND_MAX;
}
