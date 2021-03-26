/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Fri 26 Mar 2021 03:28:34 PM CST
 ************************************************************************/

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
