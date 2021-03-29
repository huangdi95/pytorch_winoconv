#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include "conv_base.cu"
#include "fused_kernels.cu"
//time measure
#include <chrono>
#define CHECK_RESULT 1
#define MY 1
#define Batch 16
//#define BN 32
//#define BC 8
//#define BK 64
//#define BN 32*32
//#define BC 8*32
#define Bi 32    //input batch
#define Hi 224  //input h
#define Wi 512 //input w
#define BC 8*8 //input c
#define BK 64   //output c
#define PH 2    //pad h
#define PW 2    //pad w

void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, int);

template<unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2DFused(const float *input, const float *weight, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int output_H, int output_W, int pad_h, int pad_w) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    int tid = by * blockDim.x + threadIdx.x;
    int bz = tid / (B* nH * nW * C);
    float accu[Batch/8][64] = {0};

    extern __shared__ float smem[]; // [16, bn, bk/2+1] with 1 conflict padding

    float *input_smem = smem;
    float *output_smem = smem + 16 * bc * bn;

    for (int i = 0; i < C; i+=bc) {
   
        //////// input transform //////
        inputNorm2WinoTransform2D_fused<bn, bc, bk>(input, input_smem, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, H, W, C, pad_h, pad_w, by, bz, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////

        float *ip = &input_smem[2*warp_id*bc*bn];
        const float *wp = &weight[2*warp_id*C*K+lane_id+i*K+bx*bk];
///////////// batched matmul bcbn 32x2x8 outer product//////////////
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
                accu[k][l] += ip[l-32] * wv;
              }
              wp += (K - 32);
              ip += bn;
          }
          wp += (C - bc) * K;
        }
        __syncthreads();
//////////////////////////////////////////////////////////////
    }
    /////////// output transform //////////////
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    bz = tid / (B* nH * nW * K);
    for (int i = 0; i < bk; i += bk/2) {
        //////// load wino output /////
        unsigned int offset = by * bn * K + bx * bk;
        for (int j = 0; j < bn; j++) {
            output_smem[((2 * warp_id) * bn + j) * (bk/2 + 1) + lane_id] = accu[0][j + i];
            output_smem[((2 * warp_id + 1) * bn + j) * (bk/2 + 1) + lane_id] = accu[1][j + i];
        }
        __syncthreads();
        //////// output transform //////
        outputWino2NormTransform2D_fused<bn, bc, bk>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, bz, warp_id, lane_id, i);
        __syncthreads();
        //////////////////////////////
    }
}



int main() {
    /****************************************************/
    /*  Preparations                                    */
    /****************************************************/
    
    int Ho=Hi-2+2*PH;
    int Wo=Wi-2+2*PW;
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
    unsigned int size_B = 3 * 3 * BC * BK;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
    float flop = 2 * (float)Bi * (float)NH * (float)NW *(float)BC * (float)BK * Batch;
    printf("wino flop: %f\n", flop);
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

//    int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6}; 
//    int H_e[] = {3, 3, 3, 6, 6, 6, 7, 7, 7}; 
//    int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6}; 
//    int W_e[] = {3, 6, 7, 3, 6, 7, 3, 6, 7}; 
//    int kernel_stride[] = {0, 16, 32, 40, 56, 72, 80, 88, 96};

    int H_s[] = {0}; 
    int H_e[] = {3}; 
    int W_s[] = {0}; 
    int W_e[] = {3}; 
    int kernel_stride[] = {0};

    const int num_split = sizeof(W_s) / sizeof(W_s[0]);
    const int N = num_split * BN * BK;

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

    cudaDeviceSynchronize();
#if MY == 1
    /****************************************************/
    /*  My kernel                                       */
    /****************************************************/

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);


    float* tmp_weight_buffer_fused;
    cudaMalloc((void**) &tmp_weight_buffer_fused, Batch*BC*BK*sizeof(float));
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    // naive implementation
    dim3 bDim2(BK, 1, 1);
    dim3 gDim2(BC, num_split, 1);
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2>>> (d_B, tmp_weight_buffer_fused, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, 3, 3, BC, BK);

    int maxbytes = 83968; // 82 KB
    cudaFuncSetAttribute(winograd2DFused<32, 8, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    winograd2DFused<32, 8, 64><<<(BN/32)*(BK/64), 256, maxbytes>>>(d_A, tmp_weight_buffer_fused, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, Ho, Wo, PH, PW);
    
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
    convLauncherStrideOneLarge2D_base<float> (d_A, d_B, tmp_input_buffer, tmp_weight_buffer, tmp_product_buffer, tmp_ptr_buffer, Bi, Hi, Wi, BC, BK, 3, 3, PH, PW, d_C, num_split, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, tmp_out_buffer, handle);
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
          if (fabs(data1[idx] - data2[idx]) > 0.00001 ) {
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
