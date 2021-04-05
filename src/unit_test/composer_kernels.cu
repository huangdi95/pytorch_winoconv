/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Thu 01 Apr 2021 04:28:23 PM CST
 ************************************************************************/
template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW, int loop_s, int loop_e>
__device__ __forceinline__ void kernel_16_gmem(const float *input, float *input_patch, float *input_smem, const float *weight, float *output_smem, float *agg_smem, const int *H_start, const int *W_start, const int *kernel_stride, const int xBase, const int yBase, const int nH, const int nW, const int B, const int H, const int W, const int C, const int K, const int output_H, const int output_W, const int bx, const int by, const int warp_id, const int lane_id, const int num_split, const int *H_end, const int *W_end) {
    float accu[16/8][bk] = {0};
    for (int count = C * loop_s; count < C * loop_e; count+=bc) {
        int bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * num_split) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
          for(int m = 0; m < (splitH + 1); m++) {
            for(int n = 0; n < (splitW + 1); n++) {
              int f_y = yBase + m + H_start[bz2];
              int f_x = xBase + n + W_start[bz2];
              if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
                input_patch[m * (splitW + 1) + n] = input[((((warp_id + i2) * H + f_y) * W + f_x)) * B + lane_id];
              } else {
                input_patch[m * (splitW + 1) + n] = float(0);
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
            output_smem[((2 * warp_id) * bn + j) * (32 + 1) + lane_id] = accu[0][j + i];
            output_smem[((2 * warp_id + 1) * bn + j) * (32 + 1) + lane_id] = accu[1][j + i];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, agg_smem, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////
    }
}

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW, int loop_s, int loop_e>
__device__ __forceinline__ void kernel_16_smem(const float *input, float *input_patch, float *input_smem, const float *weight, float *output_smem, float *agg_smem, const int *H_start, const int *W_start, const int *kernel_stride, const int xBase, const int yBase, const int nH, const int nW, const int B, const int H, const int W, const int C, const int K, const int output_H, const int output_W, const int bx, const int by, const int warp_id, const int lane_id, const int num_split, const int *H_end, const int *W_end) {
    float accu[16/8][bk] = {0};
    for (int count = C * loop_s; count < C * loop_e; count+=bc) {
        int bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * num_split) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
          for(int m = 0; m < (splitH + 1); m++) {
            for(int n = 0; n < (splitW + 1); n++) {
              int f_y = yBase + m + H_start[bz2];
              int f_x = xBase + n + W_start[bz2];
              if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
                input_patch[m * (splitW + 1) + n] = input[((((warp_id + i2) * H + f_y) * W + f_x)) * B + lane_id];
//                input_patch[m * (splitW + 1) + n] = float(0);
              } else {
                input_patch[m * (splitW + 1) + n] = float(0);
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

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW, unsigned int loop_s, unsigned int loop_e>
__global__ void winograd2DFused_16(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    int bz = loop_s;
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
    for(int j = 0; j < (splitH + 1); j++) {
      for(int k = 0; k < (splitW + 1); k++) {
        f_y = yBase + j + H_start[bz];
        f_x = xBase + k + W_start[bz];
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
          prefetch1[j * (splitW+1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
        } else {
          prefetch1[j * (splitW+1) + k] = float(0);
        }
      }
    }
/////////////////////////////////////
    for (int count = C*loop_s; count < C * loop_e; count+=bc) {
        bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * loop_e) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
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
            output_smem[((2 * warp_id) * bn + j) * (32 + 1) + lane_id] = accu[0][j + (i/32)*bn];
            output_smem[((2 * warp_id + 1) * bn + j) * (32 + 1) + lane_id] = accu[1][j + (i/32)*bn];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////
    }
}

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW, unsigned int loop_s, unsigned int loop_e>
__global__ void winograd2DFused_8(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    //TODO: when the number of warp goes down, the tiles should be less too.
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    int bz = loop_s;
    float accu[bk] = {0};

    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding

    float *input_smem = smem;
    float *output_smem = smem;

    int yBase = 2 * (by / nW) - pad_h;
    int xBase = 2 * (by % nW) - pad_w;
    int f_x, f_y;
    float prefetch1[16];
    float *input_patch = &prefetch1[0];
/////////////// prefetch /////////////
    for(int j = 0; j < (splitH + 1); j++) {
      for(int k = 0; k < (splitW + 1); k++) {
        f_y = yBase + j + H_start[bz];
        f_x = xBase + k + W_start[bz];
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
          prefetch1[j * (splitW + 1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
        } else {
          prefetch1[j * (splitW + 1) + k] = float(0);
        }
      }
    }
/////////////////////////////////////
    for (int count = C * loop_s; count < C * loop_e; count+=bc) {
        bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * loop_e) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
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
/////////////////////////////////////

        float *ip = &input_smem[warp_id * bc * bn];
        const float *wp = &weight[(warp_id + kernel_stride[bz]) * C * K + lane_id + i * K + bx * bk];
        //TODO: kernel_size[bz] when not unified
///////////// batched matmul bcbn 32x2x8 outer product//////////////
#pragma unroll
        for(int j = 0; j < bc; j++) {
            float wv = wp[0];
            for(int l = 0; l < 32; l++) {
                accu[l] += ip[l] * wv;
            }
            if (bk > 32) {
                wp += 32; 
                wv = wp[0];
                for(int l = 32; l < 64; l++) {
                    accu[l] += ip[l-32] * wv;
                }
                wp += (K - 32);
            } else if (bk <= 32) {
                wp += K;
            }
            ip += bn;
        }
        __syncthreads();
////////////////////////////////////////////////////////////////
    }
    for (int i = 0; i < bk; i += 32) {
        //////// load wino output /////
        unsigned int offset = by * bn * K + bx * bk;
        for (int j = 0; j < bn; j++) {
            output_smem[(warp_id * bn + j) * (32 + 1) + lane_id] = accu[j + (i/32)*bn];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////
    }
}

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW, unsigned int loop_s, unsigned int loop_e>
__global__ void winograd2DFused_4(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    //TODO: when the number of warp goes down, the tiles should be less too.
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    int bz = loop_s;
    float accu[bk/2] = {0};

    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding

    float *input_smem = smem;
    float *output_smem = smem;

    int yBase = 2 * (by / nW) - pad_h;
    int xBase = 2 * (by % nW) - pad_w;
    int f_x, f_y;
    float prefetch1[16];
    float *input_patch = &prefetch1[0];
/////////////// prefetch /////////////
    for(int j = 0; j < (splitH + 1); j++) {
      for(int k = 0; k < (splitW + 1); k++) {
        f_y = yBase + j + H_start[bz];
        f_x = xBase + k + W_start[bz];
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
          prefetch1[j * (splitW + 1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
        } else {
          prefetch1[j * (splitW + 1) + k] = float(0);
        }
      }
    }
/////////////////////////////////////
    for (int count = C * loop_s; count < C * loop_e; count+=bc) {
        bz = count / C;
        int i = count % C;
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
        __syncthreads();
        //////////////////////////////

/////////////// prefetch /////////////
        if (count+bc < C * loop_e) {
          int bz2 = (count + bc) / C;
          int i2 = (count + bc) % C;
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
/////////////////////////////////////

        float *ip = &input_smem[warp_id/2 * bc * bn + warp_id % 2 * 16];
        const float *wp = &weight[(warp_id/2 + kernel_stride[bz]) * C * K + lane_id + i * K + bx * bk];
        //TODO: kernel_size[bz] when not unified
///////////// batched matmul bcbn 32x2x8 outer product//////////////
#pragma unroll
        for(int j = 0; j < bc; j++) {
            float wv = wp[0];
            for(int l = 0; l < 16; l++) {
                accu[l] += ip[l] * wv;
            }
            if (bk > 32) {
                wp += 32; 
                wv = wp[0];
                for(int l = 16; l < 32; l++) {
                    accu[l] += ip[l-16] * wv;
                }
                wp += (K - 32);
            } else if (bk <= 32) {
                wp += K;
            }
            ip += bn;
        }
        __syncthreads();
////////////////////////////////////////////////////////////////
    }
    for (int i = 0; i < bk; i += 32) {
        //////// load wino output /////
        unsigned int offset = by * bn * K + bx * bk;
        for (int j = 0; j < bn/2; j++) {
            output_smem[(warp_id/2 * bn + j + (bn/2) * (warp_id%2)) * (32 + 1) + lane_id] = accu[j + (i/32)*(bn/2)];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////
    }
}

__global__ void agg(float *tmp1, float *tmp2, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size) {
    for (int i = 0; i < 4; i++) {
        tmp2[tid] += tmp1[i * size + tid];
    }
    }
}

