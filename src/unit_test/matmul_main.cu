/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Fri 19 Mar 2021 07:42:04 PM CST
 ************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil_inline.h>
//#include <helper_functions.h>

// includes, kernels
#include "matmul.cu"

////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{

    /****************************************************/
    /*  Preparations                                    */
    /****************************************************/


    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
    float flop = 2 * (float)WC * (float)HC * (float)WA;
    printf("flops: %f\n", flop);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

#if CHECK_RESULT == 1
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A, h_B, HA, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Naive CPU (Golden Reference)\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#endif

    dim3 threads,grid;

    /****************************************************/
    /*  Loop Unrolling                                  */
    /****************************************************/

    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    threads = dim3(BLOCK_SIZE, 4);
    grid = dim3(WC / (BLOCK_SIZE*4), HC / BLOCK_SIZE);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // naive implementation
    matrixMul_unroll<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Loop unrolling GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    /****************************************************/
    /*  Cleaning                                        */
    /****************************************************/

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
#if CHECK_RESULT == 1
    free(reference);
#endif
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (fabs(data1[k] - data2[k]) > 0.1 ) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf("Total Errors = %d \n", error_count);
}

