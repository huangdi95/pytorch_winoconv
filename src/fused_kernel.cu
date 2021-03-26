//#if GOOFLE_CUDA
//#define EIGEN_USE_GPU
#include <cublas_v2.h>
#include "calculation_kernels2d.cu"
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <torch/extension.h>

//#define bc 8
//#define bn 32
//#define bk 64

template <unsigned int bn, unsigned int bc, unsigned int bk>
__device__ void inputTransform2D(const float *input, float input_smem[][bc][bn], const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int pad_h, int pad_w, int N) {
    int bz = (blockIdx.x * blockDim.x + threadIdx.x) / (C * nH * nW * B); //n

    int h_end = H_end[bz];
    int h_start = H_start[bz];
    int w_end = W_end[bz];
    int w_start = W_start[bz];

    int splitxH = h_end - h_start + 1;
    int splitxW = w_end - w_start + 1;

    int xBase = 2 * (threadIdx.x / bc % nW) - pad_w;
    int yBase = 2 * (threadIdx.x / bc / nW) - pad_h;

    float input_patch[16];

    int f_x, f_y;
      for(int j = 0; j < splitxH; j++) {
        for(int k = 0; k < splitxW; k++) {
          f_y = yBase + j + h_start;
          f_x = xBase + k + w_start;
          if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[j * splitxW + k] = norm_input[(f_y * W + f_x) * C + threadIdx.x%bc];
          } else {
            input_patch[j * splitxW + k] = float(0);
          }
//          input_patch[j * splitxW + k] = float(0);
        }
      }

    float trans_input_patch[16];

    inputNorm2WinoCalculation2D(input_patch, trans_input_patch, splitxH - 1, splitxW - 1);

    for(int i = 0; i < splitxH*splitxW; i++) {
      input_smem[(i + kernel_stride[bz]) * bc * bn][threadIdx.x%bc][threadIdx.x/bc] = float(trans_input_patch[i]);
    }
}

template <unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int pad_h, int pad_w, int N)
//__global__ void winograd2D(const float *input, const float *weight, int nH, int nW, int B, int H, int W, int C, int K, int pad_h, int pad_w, int N)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    __shared__ float input_smem[16][bc][bn]; // TODO: 16 -> 100

    if(bc == 8) {
        float input[2][32];
        float accu[2][64] = {0};
    } else if(bc == 32) {
        float input[2][4];
        float accu[2][16] = {0};
    }

    for(int i = 0; i < C; i+=bc) {
   
        //////// input transform //////
        inputTransform2D<bn, bc, bk>(input, input_smem, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, H, W, C, pad_h, pad_w, N);
        //////////////////////////////
        //////// load transformed weight /////////
        // TODO
//        if(bn == 32 && bc == 8 && bk == 64) {
//            for(int j = 0; i < 16; j++) {
//                weigh_smem[i][threadIdx.x/bk][threadIdx.x%bk] = weight[i*C*K+threadIdx.x]
//                weigh_smem[i][threadIdx.x/bk+4][threadIdx.x%bk] = weight[i*C*K+threadIdx.x+256]
//            }
//        }
        //////////////////////////////////////////
        __syncthreads();
    
        ////////////// load register ////////////
        float *ip = &input_smem[warp_id*2][0][0];
        float *wp = &weight[lane_id];
        /////// batched matmul 32x2x8 outer product//////////
        // TODO
        for(int k = 0; k < 2; k++) {
          for(int j = 0; j < bc; j++) {
              float wv = wp[0];
              accu[k][0 ] += ip[0 ] * wv;
              accu[k][1 ] += ip[1 ] * wv;
              accu[k][2 ] += ip[2 ] * wv;
              accu[k][3 ] += ip[3 ] * wv;
              accu[k][4 ] += ip[4 ] * wv;
              accu[k][5 ] += ip[5 ] * wv;
              accu[k][6 ] += ip[6 ] * wv;
              accu[k][7 ] += ip[7 ] * wv;
              accu[k][8 ] += ip[8 ] * wv;
              accu[k][9 ] += ip[9 ] * wv;
              accu[k][10] += ip[10] * wv;
              accu[k][11] += ip[11] * wv;
              accu[k][12] += ip[12] * wv;
              accu[k][13] += ip[13] * wv;
              accu[k][14] += ip[14] * wv;
              accu[k][15] += ip[15] * wv;
              accu[k][16] += ip[16] * wv;
              accu[k][17] += ip[17] * wv;
              accu[k][18] += ip[18] * wv;
              accu[k][19] += ip[19] * wv;
              accu[k][20] += ip[20] * wv;
              accu[k][21] += ip[21] * wv;
              accu[k][22] += ip[22] * wv;
              accu[k][23] += ip[23] * wv;
              accu[k][24] += ip[24] * wv;
              accu[k][25] += ip[25] * wv;
              accu[k][26] += ip[26] * wv;
              accu[k][27] += ip[27] * wv;
              accu[k][28] += ip[28] * wv;
              accu[k][29] += ip[29] * wv;
              accu[k][30] += ip[30] * wv;
              accu[k][31] += ip[31] * wv;
              wp += 32; 
              wv = wp[0];
              accu[k][32] += ip[0 ] * wv;
              accu[k][33] += ip[1 ] * wv;
              accu[k][34] += ip[2 ] * wv;
              accu[k][35] += ip[3 ] * wv;
              accu[k][36] += ip[4 ] * wv;
              accu[k][37] += ip[5 ] * wv;
              accu[k][38] += ip[6 ] * wv;
              accu[k][39] += ip[7 ] * wv;
              accu[k][40] += ip[8 ] * wv;
              accu[k][41] += ip[9 ] * wv;
              accu[k][42] += ip[10] * wv;
              accu[k][43] += ip[11] * wv;
              accu[k][44] += ip[12] * wv;
              accu[k][45] += ip[13] * wv;
              accu[k][46] += ip[14] * wv;
              accu[k][47] += ip[15] * wv;
              accu[k][48] += ip[16] * wv;
              accu[k][49] += ip[17] * wv;
              accu[k][50] += ip[18] * wv;
              accu[k][51] += ip[19] * wv;
              accu[k][52] += ip[20] * wv;
              accu[k][53] += ip[21] * wv;
              accu[k][54] += ip[22] * wv;
              accu[k][55] += ip[23] * wv;
              accu[k][56] += ip[24] * wv;
              accu[k][57] += ip[25] * wv;
              accu[k][58] += ip[26] * wv;
              accu[k][59] += ip[27] * wv;
              accu[k][60] += ip[28] * wv;
              accu[k][61] += ip[29] * wv;
              accu[k][62] += ip[30] * wv;
              accu[k][63] += ip[31] * wv;
              wp += 32; 
              ip += bn;
          }
        }
        __syncthreads();
        //TODO: batch 2 has not been added; check dimension;
        ///////////////////////////////////////
    }
    ////// output transform ///////
    // TODO
    //////////////////////////////


}

int main() {
    int kernel_size = 7;
    int B = 1;
    int H = 100;
    int W = 200;
    int C = 32;
    int K = 64;
    int pad_h = 0;
    int pad_w = 0;
    int output_H =  H - kernel_size + 1 + 2 * pad_h;
    int output_W =  W - kernel_size + 1 + 2 * pad_w;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;
    int num_split = 9;
    int N = num_split*B*nH*nW*C;
    float *input_dev = nullptr;
    float *weight_dev = nullptr;
    cudaMalloc((void**)&input_dev, B*H*W*C*sizeof(float));
    cudaMalloc((void**)&weight_dev, kernel_size*kernel_size*K*C*sizeof(float));


    int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6}; 
    int H_e[] = {3, 3, 3, 6, 6, 6, 7, 7, 7}; 
    int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6}; 
    int W_e[] = {3, 6, 7, 3, 6, 7, 3, 6, 7}; 
    int kernel_stride[] = {0, 16, 32, 40, 56, 72, 80, 88, 96};

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

    float *wino_input_dev1 = nullptr;
    float *wino_input_dev2 = nullptr;
    cudaMalloc((void**)&wino_input_dev1, 100*B*nH*nW*C*sizeof(float));
    cudaMalloc((void**)&wino_input_dev2, 100*B*nH*nW*C*sizeof(float));

    winograd2D<<<(N - 256 + 1) / 256, 256>>>(input_dev, weight_dev, wino_input_dev1, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, nH, nW, B, H, W, C, K, pad_h, pad_w, N);
    
    return 0;
}
