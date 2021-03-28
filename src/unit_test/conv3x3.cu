#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include "conv_base.cu"
//time measure
#include <chrono>
#define CHECK_RESULT 1
#define MY 0
#define Batch 16
//#define BN 32
//#define BC 8
//#define BK 64
//#define BN 32*32
//#define BC 8*32
#define Bi 32    //input batch
#define Hi 128  //input h
#define Wi 128 //input w
#define BC 128 //input c
#define BK 64   //output c
#define PH 1    //pad h
#define PW 1    //pad w

void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, int);

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
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = Bi * Hi * Wi * BC;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = 3 * 3 * BC * BK;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
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
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    // naive implementation
    int maxbytes = 67584; // 96 KB
    cudaFuncSetAttribute(winograd2D<32, 8, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    winograd2D<32, 8, 64><<<(BN/32)*(BK/64), 256, maxbytes>>>(d_A, d_C, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, NH, NW, Bi, Ho, Wo, BC, BK, PH, PW);
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
//    printDiff(ref, h_C, BK, Ho, Wo, Bi);
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
