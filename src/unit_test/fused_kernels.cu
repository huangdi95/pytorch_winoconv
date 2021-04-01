/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Sun 28 Mar 2021 08:18:17 PM CST
 ************************************************************************/
#include "calculation_kernels2d.cu"
template <int splitH, int splitW>
__device__ void inputNorm2WinoTransform2D_fused(float *input_patch, float *input_smem, int warp_id, int lane_id) {

    float trans_input_patch[16];

    inputNorm2WinoCalculation2D_fused(input_patch, trans_input_patch, splitH, splitW);

//TODO: #pragma unroll
    for(int i = 0; i < (splitH + 1) * (splitW + 1); i++) {
      input_smem[i * 256 + warp_id * 32 + lane_id] = float(trans_input_patch[i]);
    }
}

template <int splitH, int splitW>
__device__ void inputNorm2WinoTransform2D_fused(float *norm_input, float *input_smem, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int pad_h, int pad_w, int by, int bz, int warp_id, int lane_id, int c_i, int h_start, int w_start) {

 //   int h_end = H_end[bz];
 //   int h_start = H_start[bz];
 //   int w_end = W_end[bz];
 //   int w_start = W_start[bz];

 //   int splitxH = h_end - h_start + 1;
 //   int splitxW = w_end - w_start + 1;

    int yBase = 2 * (by / nW) - pad_h;
    int xBase = 2 * (by % nW) - pad_w;

    float input_patch[16];

    int f_x, f_y;
//TODO: #pragma unroll
      for(int j = 0; j < (splitH+1); j++) {
        for(int k = 0; k < (splitW+1); k++) {
          f_y = yBase + j+ h_start;
          f_x = xBase + k+ w_start;
          if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
//            input_patch[j * 4 + k] = float(1);
            input_patch[j * (splitH+1) + k] = norm_input[((((warp_id + c_i) * H + f_y) * W + f_x)) * B + lane_id];
          } else {
            input_patch[j * (splitW+1) + k] = float(0);
          }
        }
      }

    float trans_input_patch[16];

    inputNorm2WinoCalculation2D_fused(input_patch, trans_input_patch, splitH, splitW);

//TODO: #pragma unroll
    for(int i = 0; i < (splitH+1)*(splitW+1); i++) {
//      input_smem[i*C*nH*nW*B+(warp_id+c_i)*nH*nW*B+by*B+lane_id] = float(trans_input_patch[i]);
      input_smem[i * 256 + warp_id * 32 + lane_id] = float(trans_input_patch[i]);
    }
}

template <unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
__device__ void outputWino2NormTransform2D_fused(float *output_smem, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int output_H, int output_W, int bx, int by, int bz, int warp_id, int lane_id, int k_i) {
    int h = by / nW; 
    int w = by % nW;

    for (int j = 0; j < 32; j += blockDim.x/32) {

        float product_patch[16] = {0};
        for (int i = 0; i < (splitH+1)*(splitW+1); i++) {
            product_patch[i] = output_smem[(i * bn + lane_id) * (32 + 1) + warp_id + j];
        }
    
        float output_patch[4] = {0};

        outputWino2NormCalculation2D_fused(product_patch, output_patch, splitH, splitW);

        unsigned int offset_k = bx * bk + k_i + warp_id + j;

        output[((offset_k * output_H + (2 * h + 0)) * output_W + (2 * w + 0)) * B + lane_id] = output_patch[0];
        if(output_W % 2 == 0 || w != nW - 1)
          output[((offset_k * output_H + (2 * h + 0)) * output_W + (2 * w + 1)) * B + lane_id] = output_patch[1];
        if(output_H % 2 == 0 || h != nH - 1)
          output[((offset_k * output_H + (2 * h + 1)) * output_W + (2 * w + 0)) * B + lane_id] = output_patch[2];
        if((output_W % 2 == 0 || w != nW - 1) && (output_H % 2 == 0 || h != nH - 1))
          output[((offset_k * output_H + (2 * h + 1)) * output_W + (2 * w + 1)) * B + lane_id] = output_patch[3];
    }
}

template <unsigned int bn, unsigned int bc, unsigned int bk>
__device__ void outputWino2NormTransform2D_fused_batch16(float *output_smem, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int output_H, int output_W, int bx, int by, int bz, int warp_id, int lane_id) {
    int h = by / nW; 
    int w = by % nW;

    int splitxH = H_end[bz] - H_start[bz] + 1;
    int splitxW = W_end[bz] - W_start[bz] + 1;

    for (int j = 0; j < 8; j++) { // bn=8, k=1

        float product_patch[16] = {1};
        for (int i = 0; i < 16; i++) {
            product_patch[i] = output_smem[i*8+j];
        }
//    
        float output_patch[4] = {0};
//
        outputWino2NormCalculation2D_fused(product_patch, output_patch, 3, 3);
//
        unsigned int offset_k = bx * bk + threadIdx.x%64;
        unsigned int offset_n = (warp_id/2)*8 + j;

        output[((offset_n * output_H + (2 * h + 0)) * output_W + (2 * w + 0)) * B + offset_k] = output_patch[0];
        if(output_W % 2 == 0 || w != nW - 1)
          output[((offset_n * output_H + (2 * h + 0)) * output_W + (2 * w + 1)) * B + offset_k] = output_patch[1];
        if(output_H % 2 == 0 || h != nH - 1)
          output[((offset_n * output_H + (2 * h + 1)) * output_W + (2 * w + 0)) * B + offset_k] = output_patch[2];
        if((output_W % 2 == 0 || w != nW - 1) && (output_H % 2 == 0 || h != nH - 1))
          output[((offset_n * output_H + (2 * h + 1)) * output_W + (2 * w + 1)) * B + offset_k] = output_patch[3];
    }
}












