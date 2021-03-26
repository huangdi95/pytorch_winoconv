/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Wed 24 Mar 2021 10:14:27 PM CST
 ************************************************************************/
template <unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int pad_h, int pad_w, int N)
//__global__ void winograd2D(const float *input, const float *weight, int nH, int nW, int B, int H, int W, int C, int K, int pad_h, int pad_w, int N)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    __shared__ float input_smem[16][bc][bn]; // TODO: 16 -> 100
    __shared__ float weigh_smem[16][bc][bk];
//    __shared__ float input_smem[16];
//    __shared__ float weight_smem[16];
    if(bc == 8) { 
        float input[2][8];
        float weigh[2][8];
        float accu[2][64] = {0};
    } else if(bc == 32) {
        float input[2][4];
        float weigh[2][4];
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
    

        for(int j = 0; j < bc; j++) {
            ////////////// load register ////////////
            if(bn == 32 && bc == 8 && bk == 64) {
                for(int k = 0; k < bc; k++) {
                    ((float4*)input[0])[k] = ((float4*)input_smem
                        [2*(warp_id)  ][k])[threadIdx.x%2+int((lane_id)/16)*2];
                    ((float4*)input[0])[k] = ((float4*)input_smem
                        [2*(warp_id)  ][k])[threadIdx.x%2+int((lane_id)/16)*2+4];
                    ((float4*)input[1])[k] = ((float4*)input_smem
                        [2*(warp_id)+1][k])[threadIdx.x%2+int((lane_id)/16)*2];
                    ((float4*)input[1])[k] = ((float4*)input_smem
                        [2*(warp_id)+1][k])[threadIdx.x%2+int((lane_id)/16)*2+4];
                    ((float4*)weigh[0])[k] = ((float4*)weigh_smem
                        [2*(warp_id)  ][k])[int((lane_id)/2)%8];
                    ((float4*)weigh[0])[k] = ((float4*)weigh_smem
                        [2*(warp_id)  ][k])[int((lane_id)/2)%8+8];
                    ((float4*)weigh[1])[k] = ((float4*)weigh_smem
                        [2*(warp_id)+1][k])[int((lane_id)/2)%8];
                    ((float4*)weigh[1])[k] = ((float4*)weigh_smem
                        [2*(warp_id)+1][k])[int((lane_id)/2)%8+8];
                }
            }
            /////// batched matmul 8x8x8 //////////
            // TODO
            
            ///////////////////////////////////////
        }
    }
    ////// output transform ///////
    // TODO
    //////////////////////////////


}
