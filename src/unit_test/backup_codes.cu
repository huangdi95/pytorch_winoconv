/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Fri 26 Mar 2021 03:28:34 PM CST
 ************************************************************************/
/************* backup batched mm with pretty pointers ****************/
template<unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D(const float *input, const float *weight, float *output, int C, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);
    float accu[Batch/8][64] = {0};
    __shared__ float input_smem[Batch][bc][bn]; // TODO: 16 -> 100

    int aBegin = by * bn * C;
    int aEnd = aBegin + C;
    int aStep = bc;

    int bBegin = bx * bk;
    int bStep = bc * K; 

    for (int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) {
//    for(int i = 0; i < C; i+=bc) {
   
        //////// input transform //////
        const float *Ap = &input[a + lane_id * C + warp_id];
#pragma unroll
        for (int j = 0; j < Batch; j++) {
//            input_smem[j][threadIdx.x%bc][threadIdx.x/bc] = input[j * bc * bn + threadIdx.x];
            input_smem[j][warp_id][lane_id] = Ap[j * C * N];
//            input_smem[j][warp_id][lane_id] = input[j * bc * bn + lane_id * bc + threadIdx.x];
        }
        //////////////////////////////
        __syncthreads();
    
        ////////////// load register ////////////
        float *ip = &input_smem[2*warp_id][0][0];
        const float *wp = &weight[2*warp_id*C*K+lane_id+b];
        /////// batched matmul 32x2x8 outer product//////////
#pragma unroll
        for(int k = 0; k < Batch/8; k++) {
#pragma unroll
          for(int j = 0; j < bc; j++) {
              float wv = wp[0];
              for(int l = 0; l < 32; l++) {
                accu[k][l] += ip[l] * wv;
              }
              wp += 32; 
              wv = wp[0];
              for(int l = 32; l < 64; l++) {
                accu[k][l] += ip[l - 32] * wv;
              }
              wp += (K - 32); 
              ip += bn;
          }
          wp += (C - bc) * K;
        }
        __syncthreads();
    }


    int cBegin = by * bn * K + bx * bk; 
#pragma unroll
    for (int i = 0; i < bn; i++) {
        output[cBegin + 2 * warp_id * N * K + i * K + lane_id] = accu[0][i];
        output[cBegin + 2 * warp_id * N * K + i * K + lane_id + 32] = accu[0][i+32];
        output[cBegin + (2 * warp_id + 1) * N * K + i * K + lane_id] = accu[1][i];
        output[cBegin + (2 * warp_id + 1) * N * K + i * K + lane_id + 32] = accu[1][i+32];
    }
}

/*********************** outer product ********************/
//              accu[k][0 ] += ip[0 ] * wv;
//              accu[k][1 ] += ip[1 ] * wv;
//              accu[k][2 ] += ip[2 ] * wv;
//              accu[k][3 ] += ip[3 ] * wv;
//              accu[k][4 ] += ip[4 ] * wv;
//              accu[k][5 ] += ip[5 ] * wv;
//              accu[k][6 ] += ip[6 ] * wv;
//              accu[k][7 ] += ip[7 ] * wv;
//              accu[k][8 ] += ip[8 ] * wv;
//              accu[k][9 ] += ip[9 ] * wv;
//              accu[k][10] += ip[10] * wv;
//              accu[k][11] += ip[11] * wv;
//              accu[k][12] += ip[12] * wv;
//              accu[k][13] += ip[13] * wv;
//              accu[k][14] += ip[14] * wv;
//              accu[k][15] += ip[15] * wv;
//              accu[k][16] += ip[16] * wv;
//              accu[k][17] += ip[17] * wv;
//              accu[k][18] += ip[18] * wv;
//              accu[k][19] += ip[19] * wv;
//              accu[k][20] += ip[20] * wv;
//              accu[k][21] += ip[21] * wv;
//              accu[k][22] += ip[22] * wv;
//              accu[k][23] += ip[23] * wv;
//              accu[k][24] += ip[24] * wv;
//              accu[k][25] += ip[25] * wv;
//              accu[k][26] += ip[26] * wv;
//              accu[k][27] += ip[27] * wv;
//              accu[k][28] += ip[28] * wv;
//              accu[k][29] += ip[29] * wv;
//              accu[k][30] += ip[30] * wv;
//              accu[k][31] += ip[31] * wv;
//              wp += 32; 
//              wv = wp[0];
//              accu[k][32] += ip[0 ] * wv;
//              accu[k][33] += ip[1 ] * wv;
//              accu[k][34] += ip[2 ] * wv;
//              accu[k][35] += ip[3 ] * wv;
//              accu[k][36] += ip[4 ] * wv;
//              accu[k][37] += ip[5 ] * wv;
//              accu[k][38] += ip[6 ] * wv;
//              accu[k][39] += ip[7 ] * wv;
//              accu[k][40] += ip[8 ] * wv;
//              accu[k][41] += ip[9 ] * wv;
//              accu[k][42] += ip[10] * wv;
//              accu[k][43] += ip[11] * wv;
//              accu[k][44] += ip[12] * wv;
//              accu[k][45] += ip[13] * wv;
//              accu[k][46] += ip[14] * wv;
//              accu[k][47] += ip[15] * wv;
//              accu[k][48] += ip[16] * wv;
//              accu[k][49] += ip[17] * wv;
//              accu[k][50] += ip[18] * wv;
//              accu[k][51] += ip[19] * wv;
//              accu[k][52] += ip[20] * wv;
//              accu[k][53] += ip[21] * wv;
//              accu[k][54] += ip[22] * wv;
//              accu[k][55] += ip[23] * wv;
//              accu[k][56] += ip[24] * wv;
//              accu[k][57] += ip[25] * wv;
//              accu[k][58] += ip[26] * wv;
//              accu[k][59] += ip[27] * wv;
//              accu[k][60] += ip[28] * wv;
//              accu[k][61] += ip[29] * wv;
//              accu[k][62] += ip[30] * wv;
//              accu[k][63] += ip[31] * wv;

/****************************************************/
/*  bn=64, bc=4, bn=64,                             */
/*  1847/5392 GFLOPS when C=64,                     */
/*  1004/1156 GFLOPS when C=4                       */
/****************************************************/
template<unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D_bcbn_batch16(const float *input, const float *weight, float *output, int C, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);
    float accu[Batch*8*2] = {0};
    __shared__ float input_smem[Batch*bc*bn]; // TODO: 16 -> 100
//    __shared__ float weight_smem[Batch*bc*bk]; // TODO: 16 -> 100

    for(int i = 0; i < C; i+=bc) {
   
        //////// input transform //////
#pragma unroll
        for (int j = 0; j < Batch; j++) {
//            input_smem[j][threadIdx.x%bc][threadIdx.x/bc] = input[j * bc * bn + threadIdx.x];
            input_smem[j*256+threadIdx.x] = input[j * C * N + by * C * bn + threadIdx.x%64 * C + i + threadIdx.x/64];
//            weight_smem[j*512+warp_id*64+lane_id] = weight[j * C * K + (warp_id + i) * K + lane_id + bx*bk];
//            weight_smem[j*512+warp_id*64+lane_id+32] = weight[j * C * K + (warp_id + i) * K + lane_id + 32 + bx*bk];
//            input_smem[j][warp_id][lane_id] = input[j * bc * bn + lane_id * bc + threadIdx.x];
        }
        //////////////////////////////
        __syncthreads();
    
        ////////////// load register ////////////
        float *ip = &input_smem[8*warp_id];
        const float *wp = &weight[lane_id+i*K+bx*bk];
//        float *wp = &weight_smem[lane_id];
        /////// batched matmul 32x2x8 outer product//////////
//#pragma unroll
//        for(int k = 0; k < Batch/8; k++) {
//#pragma unroll
//          for(int j = 0; j < bc; j++) {
//              float wv = wp[0];
//              for(int l = 0; l < 32; l++) {
//                accu[k*64+l] += ip[l] * wv;
//              }
//              wp += 32; 
//              wv = wp[0];
//              for(int l = 32; l < 64; l++) {
//                accu[k*64+l] += ip[l-32] * wv;
//              }
//              wp += (K - 32); 
//              ip += bn;
//          }
//          wp += (C - bc) * K;
//        }
#pragma unroll
        for(int k = 0; k < Batch; k++) {
#pragma unroll
          for(int j = 0; j < bc; j++) {
              float wv = wp[0];
              for(int l = 0; l < 8; l++) {
                accu[k*16+l] += ip[l] * wv;
              }
              wp += 32;
              wv = wp[0];
              for(int l = 8; l < 16; l++) {
                accu[k*16+l] += ip[l-8] * wv;
              }
              wp += (K - 32);
              ip += bn;
          }
          wp += (C - bc) * K;
        }
//#pragma unroll
//        for(int k = 0; k < Batch; k++) {
//#pragma unroll
//          for(int j = 0; j < bc; j++) {
//              float wv1 = wp[0];
//              float wv2 = wp[32];
//              for(int l = 0; l < 4; l++) {
//                accu[k][l] += ip[l] * wv1;
//                accu[k][l+4] += ip[l] * wv2;
//              }
//              wp += K; 
//              ip += bn;
//          }
//          wp += (C - bc) * K;
//        }
        __syncthreads();
    }
    unsigned int offset = (by * bn + warp_id * 8) * K + bx * bk;
#pragma unroll
    //TODO: need smem padding when composed
//    for (int i = 0; i < bn; i++) {
//        output[offset + 2 * warp_id * N * K + i * K + lane_id] = accu[i];
//        output[offset + 2 * warp_id * N * K + i * K + lane_id + 32] = accu[i+32];
//        output[offset + (2 * warp_id + 1) * N * K + i * K + lane_id] = accu[64+i];
//        output[offset + (2 * warp_id + 1) * N * K + i * K + lane_id + 32] = accu[64+i+32];
//    }
    for (int i = 0; i < Batch; i++) {
        for (int j = 0; j < 8; j++) {
            output[offset + i * N * K + j * K + lane_id] = accu[i*16 + j];
            output[offset + i * N * K + j * K + lane_id + 32] = accu[i*16 + j + 8];
//            output[offset + i * N * K + j * K + lane_id + 32] = accu[i*64 + j + 32];
        }
    }
}
