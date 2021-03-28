/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Fri 26 Mar 2021 04:58:38 PM CST
 ************************************************************************/
#include<iostream>
#include<stdio.h>
#include "../calculation_kernels2d.cu"
using namespace std;
#define CHECK_RESULT 1
#define MY 1
#define Batch 16
//#define BN 32
//#define BC 8
//#define BK 64
//#define BN 32*32
//#define BC 8*32
#define Bi 32    //input batch
//#define Hi 448  //input h
//#define Wi 1024 //input w
#define Hi 224  //input h
#define Wi 512 //input w
#define BC 8 //input c
#define BK 64   //output c
#define PH 2    //pad h
#define PW 2    //pad w

void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, int);

template <typename T>
__global__ void outputWino2NormTransform2D_base(const T *wino_output, T *tmp_output, const int *kernel_stride,  const int *H_start, const int *H_end, const int *W_start, const int *W_end, int B, int output_H, int output_W, int K, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
   int nH, nW;
   nH = (output_H + 1) / 2;
   nW = (output_W + 1) / 2;
   int bz = tid / (K * nH * nW * B); //n
   int by = (tid % (K * nH * nW * B)) / (K * nH * nW); //b
   int bx = (tid % (K * nH * nW * B) % (K * nH * nW)) / K; //h*w
   int tx = tid % (K * nH * nW * B) % (K * nH * nW) % K; //K

   int h = bx / nW; 
   int w = bx % nW;

   int splitxH = H_end[bz] - H_start[bz] + 1;
   int splitxW = W_end[bz] - W_start[bz] + 1;

    T product_patch[16] = {0};

    product_patch[0] = wino_output[((((0 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[1] = wino_output[((((1 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[2] = wino_output[((((2 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[3] = wino_output[((((3 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[4] = wino_output[((((4 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[6] = wino_output[((((6 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[7] = wino_output[((((7 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[8] = wino_output[((((8 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[9] = wino_output[((((9 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[10] = wino_output[((((10 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[11] = wino_output[((((11 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[12] = wino_output[((((12 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[13] = wino_output[((((13 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[14] = wino_output[((((14 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];
    product_patch[15] = wino_output[((((15 + kernel_stride[bz]) * B + by) * nH + h) * nW + w) * K + tx];

    T output_patch[4] = {0};

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
}

template <unsigned int bn, unsigned int bc, unsigned int bk>
__device__ void outputWino2NormTransform2D(float *output_smem, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int output_H, int output_W, int bx, int by, int bz, int warp_id, int lane_id, int k_i) {
    int h = by / nW; 
    int w = by % nW;

    int splitxH = H_end[bz] - H_start[bz] + 1;
    int splitxW = W_end[bz] - W_start[bz] + 1;

    float product_patch[16] = {0};

    for (int i = 0; i < splitxH*splitxW; i++) {
        for (int j = 0; j < 32; j += 8) {
            product_patch[i] = output_smem[(i * bn + lane_id) * (bk/2 + 1) + warp_id + j];
    
            float output_patch[4] = {0};

            outputWino2NormCalculation2D(product_patch, output_patch, splitxH - 1, splitxW - 1);

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
}

template<unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D(const float *wino_output, float *output, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int output_H, int output_W, int C, int K, int pad_h, int pad_w)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);  //TODO: slow???
    int tid = by * blockDim.x + threadIdx.x;
    int bz = tid / (B* nH * nW * C);
    float accu[Batch/8][64] = {0};

//    __shared__ float output_smem[16 * bn * (bk/2 + 1)]; // [16, bn, bk/2+1] with 1 conflict padding
    extern __shared__ float output_smem[]; // [16, bn, bk/2+1] with 1 conflict padding

    unsigned int offset_k = bx * bk;
    unsigned int offset_n = by * bn;
    for (int i = 0; i < bk; i += bk/2) {
        //////// load wino output /////
        for (int j = 0; j < 32; j++) {
            output_smem[((2 * warp_id) * bn + j) * (bk/2 + 1) + lane_id] = wino_output[((2 * warp_id) * B * nH * nW + (j + offset_n)) * K + lane_id + offset_k + i];
            output_smem[((2 * warp_id + 1) * bn + j) * (bk/2 + 1) + lane_id] = wino_output[((2 * warp_id + 1) * B * nH * nW + (j + offset_n)) * K + lane_id + offset_k + i];
        }
        __syncthreads();
        //////// input transform //////
        outputWino2NormTransform2D<bn, bc, bk>(output_smem, output, kernel_stride, H_start, H_end, W_start, W_end, nH, nW, B, output_H, output_W, bx, by, bz, warp_id, lane_id, i);
        //////////////////////////////
    }
}

int main() {
    /****************************************************/
    /*  Preparations                                    */
    /****************************************************/
    
    int Ho=Hi-2+PH;
    int Wo=Wi-2+PW;
    int NH=(Hi-2+PH)/2; //nH
    int NW=(Wi-2+PW)/2; //nW
    int BN=NH*NW*Bi;  //N 

    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = Batch * Bi * NH * NW * BK;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    // initialize host memory
    randomInit(h_A, size_A);
    
    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);

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
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    // naive implementation
    int maxbytes = 67584; // 96 KB
    cudaFuncSetAttribute(winograd2D<32, 8, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    winograd2D<32, 8, 64><<<(BN/32)*(BK/64), 256, 67584>>>(d_A, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Ho, Wo, BC, BK, PH, PW);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("My\n");
    printf("Processing time: %f (ms) \n", msecTotal);
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

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
////////////TODO: transform ///////////////
    outputWino2NormTransform2D_base <float> <<<(N - 1 + 512) / 512, 512>>> (d_A, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, Bi, Ho, Wo, BK, N);
///////////////////////////////////////////
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Baseline\n");
    printf("Processing time: %f (ms) \n", msecTotal);
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
    printDiff(ref, h_C, Batch, NH, NW, BC);
    free(ref);
#endif
    free(h_A);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_C);
    return 0;
}

void printDiff(float *data1, float *data2, int B, int nH, int nW, int C)
{
  int i,j,idx,l,x;
  int error_count=0;
    
  for (l=0; l<B; l++) {
    for (j=0; j<nH; j++) {
      for (i=0; i<nW; i++) {
        for (x=0; x<C; x++) {
          idx = l*nH*nW*C+j*nW*C+i*C+x;
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
