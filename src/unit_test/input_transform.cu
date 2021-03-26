/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Fri 26 Mar 2021 04:58:38 PM CST
 ************************************************************************/
#include<iostream>
#include<stdio.h>
#include<cublas_v2.h>
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
#define Bi 1    //input batch
//#define Hi 448  //input h
//#define Wi 1024 //input w
#define Hi 8*32  //input h
#define Wi 16*32 //input w
#define BC 8*4   //input c
#define BK 64*2   //output c
#define PH 2    //pad h
#define PW 2    //pad w
#define NH (Hi-2+PH)/2 //nH
#define NW (Wi-2+PW)/2 //nW
#define BN NH*NW*Bi  //N 

void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, int);

template <typename T>
__global__ void inputNorm2WinoTransform2D_base(const T *norm_input, T *wino_input, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int pad_h, int pad_w, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
    int bz = tid / (C * nH * nW * B); //n
    int by = (tid % (C * nH * nW * B)) / (C * nH * nW); //b
    int bx = (tid % (C * nH * nW * B) % (C * nH * nW)) / C; //h*w
    int tx = tid % (C * nH * nW * B) % (C * nH * nW) % C; //K

    int h = bx / nW; 
    int w = bx % nW;

    int h_end = H_end[bz];
    int h_start = H_start[bz];
    int w_end = W_end[bz];
    int w_start = W_start[bz];

    int splitxH = h_end - h_start + 1;
    int splitxW = w_end - w_start + 1;

    int f_b = by;
    int xBase = 2 * w - pad_w;
    int yBase = 2 * h - pad_h;

    T input_patch[16];

    int f_x, f_y;
      for(int j = 0; j < splitxH; j++) {
        for(int k = 0; k < splitxW; k++) {
          f_y = yBase + j + h_start;
          f_x = xBase + k + w_start;
          if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[j * splitxW + k] = norm_input[((f_b * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[j * splitxW + k] = T(0);
          }
        }
      }

    T trans_input_patch[16];

    inputNorm2WinoCalculation2D(input_patch, trans_input_patch, splitxH - 1, splitxW - 1);

    int offset = ((f_b * nH + h) * nW + w) * C + tx;
    int stride = B * nH * nW * C;

    for(int i = 0; i < splitxH*splitxW; i++) {
      wino_input[(i + kernel_stride[bz]) * stride + offset] = T(trans_input_patch[i]);
    }
    }
}

template <unsigned int bn, unsigned int bc, unsigned int bk>
__device__ void inputNorm2WinoTransform2D(const float *norm_input, float input_smem[][bc*bn], const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nW, int H, int W, int C, int pad_h, int pad_w, int bx, int by, int bz, int warp_id, int lane_id, int c_i) {

    int h_end = H_end[bz];
    int h_start = H_start[bz];
    int w_end = W_end[bz];
    int w_start = W_start[bz];

    int splitxH = h_end - h_start + 1;
    int splitxW = w_end - w_start + 1;

    int yBase = 2 * ((by*bn + lane_id) / nW) - pad_h;
    int xBase = 2 * ((by*bn + lane_id) % nW) - pad_w;

    float input_patch[16];

    int f_x, f_y;
      for(int j = 0; j < splitxH; j++) {
        for(int k = 0; k < splitxW; k++) {
          f_y = yBase + j + h_start;
          f_x = xBase + k + w_start;
          if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[j * splitxW + k] = norm_input[(f_y * W + f_x) * C + threadIdx.x/32 + c_i];
            //TODO: NO Batch supported; threadIdx.x/32
          } else {
            input_patch[j * splitxW + k] = float(0);
          }
//          input_patch[j * splitxW + k] = float(0);
        }
      }

    float trans_input_patch[16];

    inputNorm2WinoCalculation2D(input_patch, trans_input_patch, splitxH - 1, splitxW - 1);

    for(int i = 0; i < splitxH*splitxW; i++) {
      input_smem[i][warp_id*32+lane_id] = float(trans_input_patch[i]);
    }
}

template<unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D(const float *input, float *wino_input, const int *kernel_stride, const int *H_start, const int *H_end, const int *W_start, const int *W_end, int nH, int nW, int B, int H, int W, int C, int K, int pad_h, int pad_w)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bz = tid / (B* nH * nW * C);
//    float accu[Batch/8][64] = {0};

    __shared__ float input_smem[16][bc*bn]; // TODO: 16 -> 100

    for (int i = 0; i < C; i+=bc) {
   
        //////// input transform //////
        inputNorm2WinoTransform2D<bn, bc, bk>(input, input_smem, kernel_stride, H_start, H_end, W_start, W_end, nW, H, W, C, pad_h, pad_w, bx, by, bz, warp_id, lane_id, i);
        //////////////////////////////
        for (int j = 0; j < 16; j++) {
            wino_input[j*nH*nW*C+by*bn*C+lane_id*C+warp_id+i] = input_smem[j][warp_id*32+lane_id];
        }  
    }
}

int main() {
    /****************************************************/
    /*  Preparations                                    */
    /****************************************************/

    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = Bi * Hi * Wi * BC;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    // initialize host memory
    randomInit(h_A, size_A);
    
    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);

    // allocate device memory for result
    unsigned int size_C = Batch * BN * BC;
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
    const int N = BN * BC;

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
    winograd2D<32, 8, 64><<<(BN/32)*(BK/64), 256>>>(d_A, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, BK, PH, PW);
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
    inputNorm2WinoTransform2D_base <float> <<<(N - 1 + 512) / 512, 512>>> (d_A, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Hi, Wi, BC, PH, PW, N);
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
