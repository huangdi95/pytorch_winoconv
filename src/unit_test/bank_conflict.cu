/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Fri 26 Mar 2021 10:30:48 AM CST
 ************************************************************************/
#include <stdio.h>

#define CHECK_RESULT 1
// Thread block size
#define BLOCK_SIZE 16

// outer product vetor size is VECTOR_SIZE * BLOCK_SIZE
#define VECTOR_SIZE 4

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
//#define WA 4*BLOCK_SIZE // Matrix A width
//#define HA 4*BLOCK_SIZE // Matrix A height
//#define WB 4*BLOCK_SIZE // Matrix B width

#define WA 32*BLOCK_SIZE // Matrix A width
#define HA 16*BLOCK_SIZE // Matrix A height
#define WB 24*BLOCK_SIZE // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif


void randomInit(float*, int);
void printDiff(float*, float*, int, int);
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
__global__ void
matrixMul_noBankConflict( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {


        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
      Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

int main() {
    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
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
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

    dim3 threads,grid;
    /****************************************************/
    /*  Global mem coalescing w/o smem bank conflict    */
    /****************************************************/
    // setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // naive implementation
    matrixMul_noBankConflict<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);


#if CHECK_RESULT == 1
    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A, h_B, HA, WA, WB);
    printDiff(reference, h_C, WC, HC);
    free(reference);
#endif
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

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
// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}
