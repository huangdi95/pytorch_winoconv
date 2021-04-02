/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Thu 01 Apr 2021 04:28:23 PM CST
 ************************************************************************/
template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
__global__ void winograd2DFused_16(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    float accu[16/8][bk] = {0};

    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding

    float *input_smem = smem;
    float *output_smem = smem + 16 * bc * bn;

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
        inputNorm2WinoTransform2D_fused<splitH, splitW>(input_patch, input_smem, warp_id, lane_id);
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

template<unsigned int bn, unsigned int bc, unsigned int bk, int splitH, int splitW>
__global__ void winograd2DFused_8(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w, int num_split) {
    //TODO: when the number of warp goes down, the tiles should be less too.
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    float accu[bk] = {0};

    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding

    float *input_smem = smem;
    float *output_smem = smem + (splitH + 1) * (splitW + 1) * bc * bn;

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
          prefetch1[j * (splitW + 1) + k] = input[((((warp_id + 0) * H + f_y) * W + f_x)) * B + lane_id];
        } else {
          prefetch1[j * (splitW + 1) + k] = float(0);
        }
      }
    }
/////////////////////////////////////
    for (int count = 0; count < C * num_split; count+=bc) {
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
        const float *wp = &weight[(warp_id + (splitH + 1) * (splitW + 1) * bz) * C * K + lane_id + i * K + bx * bk];
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
            output_smem[(warp_id * bn + j) * (32 + 1) + lane_id] = accu[j + i];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused<bn, bc, bk, splitH, splitW>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, 0, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////
    }
}
