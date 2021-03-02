/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Thu 26 Nov 2020 04:16:33 PM CST
 ************************************************************************/
//#if GOOFLE_CUDA
//#define EIGEN_USE_GPU
#include <cublas_v2.h>
#include "calculation_kernels.cu"
// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (27, C, K)
// wino_weight = (64, C, K)
template <typename T>
__global__ void wNorm2WinoTransform(const T *norm_weight, T* wino_weight, const int *kernel_stride, const int *D_start, const int *D_end, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int D, int H, int W, int C, int K)
{

    int by = blockIdx.y; // n
    int c = blockIdx.x; // c
    int k = threadIdx.x; // k
//    if(by*c+k == 0)
//        printf("wNorm2WinoTransform called!!!!!!!!!!!!!!!!!\n");

    int splitD = D_end[by] - D_start[by];
    int splitH = H_end[by] - H_start[by];
    int splitW = W_end[by] - W_start[by];
    
//    TODO: need to +1
    T ele[27];

    for(int d = 0; d < splitD; d++) {
        for(int h = 0; h < splitH; h++) {
            for(int w = 0; w < splitW; w++) {
                ele[(d * splitH + h) * splitW + w] = norm_weight[((((d + D_start[by]) * H + h + H_start[by]) * W + w + W_start[by]) * C + c) * K + k];
            }
        }
    }

    T product_weight_patch[64];

//////// TODO: transformation functions here /////////////
//
//  __device__ function();
    wNorm2WinoCalculation(ele, product_weight_patch, splitD, splitH, splitW);
//
//////////////////////////////////////////////////////////


    for(int i = 0; i < (splitD+1)*(splitH+1)*(splitW+1); i++) {
//        product_weight_patch[i] = ele[i];
        wino_weight[(i + kernel_stride[by]) * C * K + c * K + k] = product_weight_patch[i];
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (64, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform(const T *norm_input, T *wino_input, const int *kernel_stride, const int *D_start, const int *D_end, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nD, int nH, int nW, int B, int D, int H, int W, int C, int pad_d, int pad_h, int pad_w) {
    int bz = blockIdx.z; //n
    int by = blockIdx.y; //b
    int bx = blockIdx.x; //d*h*w
    int tx = threadIdx.x; //K
//    if(by*bx+tx == 0)
//    printf("inputNorm2WinoTransform called!!!!!!!!!!!!!!!!!\n");

    int d = bx / (nH * nW); 
    int h = (bx % (nH * nW)) / nW; 
    int w = (bx % (nH * nW)) % nW;

    int splitxD = D_end[bz] - D_start[bz] + 1;
    int splitxH = H_end[bz] - H_start[bz] + 1;
    int splitxW = W_end[bz] - W_start[bz] + 1;

    int f_b = by;
    int zBase = 2 * d - pad_d;
    int xBase = 2 * w - pad_w;
    int yBase = 2 * h - pad_h;


    T input_patch[64] = {T(0)};
//    T *input_patch = new T[splitxD*splitxH*splitxW];

    int f_x, f_y, f_z;
    for(int i = 0; i < splitxD; i++) {
      for(int j = 0; j < splitxH; j++) {
        for(int k = 0; k < splitxW; k++) {
          f_z = zBase + i + D_start[bz];
          f_y = yBase + j + H_start[bz];
          f_x = xBase + k + W_start[bz];
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[(i * splitxH + j) * splitxW + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[(i * splitxH + j) * splitxW + k] = T(0);
          }
        }
      }
    }

//    T *trans_input_patch = new T[splitxD*splitxH*splitxW];
    T trans_input_patch[64] = {T(0)};

//////// TODO: transformation functions here /////////////
//
//  __device__ function();
    inputNorm2WinoCalculation(input_patch, trans_input_patch, splitxD - 1, splitxH - 1, splitxW - 1);
//
//////////////////////////////////////////////////////////

    int offset =  (((f_b * nD + d) * nH + h) * nW + w) * C + tx;
    int stride = B * nD * nH * nW * C;

    for(int i = 0; i < splitxD*splitxH*splitxW; i++) {
//      trans_input_patch[i] = input_patch[i];
      wino_input[(i + kernel_stride[bz]) * stride + offset] = T(trans_input_patch[i]);
//      wino_input[(i + kernel_stride[bz]) * stride + offset] = 1;
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (64, Batch, nD, nH, nW, K)
//tmp_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform(const T *wino_output, T *tmp_output, const int *kernel_stride, const int *D_start, const int *D_end, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //n
    int by = blockIdx.y; //b
    int bx = blockIdx.x; //d*h*w
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;
//    if(by*bx+tx == 0)
//    printf("outputWino2NormTransform called!!!!!!!!!!!!!!!!!\n");

    int d = bx / (nH * nW); 
    int h = (bx % (nH * nW)) / nW; 
    int w = (bx % (nH * nW)) % nW;

    int splitxD = D_end[bz] - D_start[bz] + 1;
    int splitxH = H_end[bz] - H_start[bz] + 1;
    int splitxW = W_end[bz] - W_start[bz] + 1;

    T product_patch[64] = {T(0)};

    for(int i = 0; i < splitxD*splitxH*splitxW; i++) {
      product_patch[i] = wino_output[(((((i + kernel_stride[bz]) * B + by) * nD + d) * nH + h) * nW + w) * K + tx];
    }

    T output_patch[8] = {T(0)};

//////// TODO: transformation functions here /////////////
//
//  __device__ function();
    outputWino2NormCalculation(product_patch, output_patch, splitxD - 1, splitxH - 1, splitxW - 1);
//
//////////////////////////////////////////////////////////

    tmp_output[((((bz * B + by) * output_D + (2 * d + 0)) * output_H + (2 * h + 0)) * output_W + (2 * w + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || w != nW - 1)
      tmp_output[((((bz * B + by) * output_D + (2 * d + 0)) * output_H + (2 * h + 0)) * output_W + (2 * w + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || h != nH - 1)
      tmp_output[((((bz * B + by) * output_D + (2 * d + 0)) * output_H + (2 * h + 1)) * output_W + (2 * w + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || w != nW - 1) && (output_H % 2 == 0 || h != nH - 1))
      tmp_output[((((bz * B + by) * output_D + (2 * d + 0)) * output_H + (2 * h + 1)) * output_W + (2 * w + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || d != nD - 1)
      tmp_output[((((bz * B + by) * output_D + (2 * d + 1)) * output_H + (2 * h + 0)) * output_W + (2 * w + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || d != nD - 1) && (output_W % 2 == 0 || w != nW - 1))
      tmp_output[((((bz * B + by) * output_D + (2 * d + 1)) * output_H + (2 * h + 0)) * output_W + (2 * w + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || d != nD - 1) && (output_H % 2 == 0 || h != nH - 1))
      tmp_output[((((bz * B + by) * output_D + (2 * d + 1)) * output_H + (2 * h + 1)) * output_W + (2 * w + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || d != nD - 1) && (output_W % 2 == 0 || w != nW - 1 ) && (output_H % 2 == 0 || h != nH - 1))
      tmp_output[((((bz * B + by) * output_D + (2 * d + 1)) * output_H + (2 * h + 1)) * output_W + (2 * w + 1)) * K + tx] = output_patch[7];
    
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//tmp_output = (64, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputAggregate(T *tmp_output, T *norm_output, int numSplit, int B, int output_D, int output_H, int output_W, int K) {
    int by = blockIdx.y; //b
    int bx = blockIdx.x; //d*h*w
    int tx = threadIdx.x; //K
//    if(by*bx+tx == 0)
//    printf("outputAggregate called!!!!!!!!!!!!!!!!!\n");

    int d = bx / (output_H * output_W);
    int h = (bx % (output_H * output_W)) / output_W;
    int w = (bx % (output_H * output_W)) % output_W;

    T result = (T)0;

    for(int i = 0; i < numSplit; i++) {
      result += tmp_output[((((i * B + by) * output_D + d) * output_H + h) * output_W + w) * K + tx];
    }

    norm_output[(((by * output_D + d) * output_H + h) * output_W + w) * K + tx] = result;
}
