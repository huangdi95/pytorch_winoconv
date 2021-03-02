/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Thu 26 Nov 2020 04:16:33 PM CST
 ************************************************************************/
//#if GOOFLE_CUDA
//#define EIGEN_USE_GPU
#include <cublas_v2.h>
#include "calculation_kernels2d.cu"
// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (27, C, K)
// wino_weight = (64, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2D(const T *norm_weight, T* wino_weight, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int H, int W, int C, int K)
{

    int by = blockIdx.y; // n
    int c = blockIdx.x; // c
    int k = threadIdx.x; // k

    int splitH = H_end[by] - H_start[by];
    int splitW = W_end[by] - W_start[by];
    
//    TODO: need to +1
    T ele[9];

    for(int h = 0; h < splitH; h++) {
        for(int w = 0; w < splitW; w++) {
            ele[h * splitW + w] = norm_weight[(((h + H_start[by]) * W + w + W_start[by]) * C + c) * K + k];
        }
    }

    T product_weight_patch[16];

    wNorm2WinoCalculation2D(ele, product_weight_patch, splitH, splitW);


    for(int i = 0; i < (splitH+1)*(splitW+1); i++) {
//        product_weight_patch[i] = ele[i];
        wino_weight[(i + kernel_stride[by]) * C * K + c * K + k] = product_weight_patch[i];
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (64, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2D(const T *norm_input, T *wino_input, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int pad_h, int pad_w) {
    int bz = blockIdx.z; //n
    int by = blockIdx.y; //b
    int bx = blockIdx.x; //h*w
    int tx = threadIdx.x; //K
//    if(by*bx+tx == 0)
//    printf("inputNorm2WinoTransform called!!!!!!!!!!!!!!!!!\n");

    int h = bx / nW; 
    int w = bx % nW;

    int splitxH = H_end[bz] - H_start[bz] + 1;
    int splitxW = W_end[bz] - W_start[bz] + 1;

    int f_b = by;
    int xBase = 2 * w - pad_w;
    int yBase = 2 * h - pad_h;


    T input_patch[16] = {T(0)};
//    T *input_patch = new T[splitxD*splitxH*splitxW];

    int f_x, f_y;
      for(int j = 0; j < splitxH; j++) {
        for(int k = 0; k < splitxW; k++) {
          f_y = yBase + j + H_start[bz];
          f_x = xBase + k + W_start[bz];
          if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[j * splitxW + k] = norm_input[((f_b * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[j * splitxW + k] = T(0);
          }
        }
      }

//    T *trans_input_patch = new T[splitxD*splitxH*splitxW];
    T trans_input_patch[16] = {T(0)};

//////// TODO: transformation functions here /////////////
//
//  __device__ function();
    inputNorm2WinoCalculation2D(input_patch, trans_input_patch, splitxH - 1, splitxW - 1);
//
//////////////////////////////////////////////////////////

    int offset = ((f_b * nH + h) * nW + w) * C + tx;
    int stride = B * nH * nW * C;

    for(int i = 0; i < splitxH*splitxW; i++) {
      wino_input[(i + kernel_stride[bz]) * stride + offset] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (64, Batch, nD, nH, nW, K)
//tmp_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2D(const T *wino_output, T *tmp_output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int B, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //n
    int by = blockIdx.y; //b
    int bx = blockIdx.x; //h*w
    int tx = threadIdx.x; //K
    int nH, nW;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;

    int h = bx / nW; 
    int w = bx % nW;

    int splitxH = H_end[bz] - H_start[bz] + 1;
    int splitxW = W_end[bz] - W_start[bz] + 1;

    T product_patch[16] = {T(0)};

    for(int i = 0; i < splitxH*splitxW; i++) {
      product_patch[i] = wino_output[((((i + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    }

    T output_patch[4] = {T(0)};

//////// TODO: transformation functions here /////////////
//
//  __device__ function();
    outputWino2NormCalculation2D(product_patch, output_patch, splitxH - 1, splitxW - 1);
//
//////////////////////////////////////////////////////////

    tmp_output[(((bz * B + by) * output_H + (2 * h + 0)) * output_W + (2 * w + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || w != nW - 1)
      tmp_output[(((bz * B + by) * output_H + (2 * h + 0)) * output_W + (2 * w + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || h != nH - 1)
      tmp_output[(((bz * B + by) * output_H + (2 * h + 1)) * output_W + (2 * w + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || w != nW - 1) && (output_H % 2 == 0 || h != nH - 1))
      tmp_output[(((bz * B + by) * output_H + (2 * h + 1)) * output_W + (2 * w + 1)) * K + tx] = output_patch[3];
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//tmp_output = (64, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputAggregate2D(T *tmp_output, T *norm_output, int numSplit, int B, int output_H, int output_W, int K) {
    int by = blockIdx.y; //b
    int bx = blockIdx.x; //h*w
    int tx = threadIdx.x; //K
//    if(by*bx+tx == 0)
//    printf("outputAggregate called!!!!!!!!!!!!!!!!!\n");

    int h = bx / output_W;
    int w = bx % output_W;

    T result = (T)0;

    for(int i = 0; i < numSplit; i++) {
      result += tmp_output[(((i * B + by) * output_H + h) * output_W + w) * K + tx];
    }

    norm_output[((by * output_H + h) * output_W + w) * K + tx] = result;
}
