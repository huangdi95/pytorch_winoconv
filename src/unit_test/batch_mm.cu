/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Thu 25 Mar 2021 03:10:24 PM CST
 ************************************************************************/
#include<iostream>
#include<stdio.h>
#include<cublas_v2.h>
using namespace std;
#define CHECK_RESULT 0
#define CUBLAS 1
#define MY_bcbn 0
#define MY_bnbc 1
////////TODO: not correct yet
#define Batch 16

#define BN 32
#define BC 8
#define BK 64

//#define BN 114688
//#define BC 64
//#define BK 64

void randomInit(float*, int);
void printDiff(float*, float*, int, int);
void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int l = 0; l < Batch; ++l)
        for (unsigned int i = 0; i < hA; ++i)
            for (unsigned int j = 0; j < wB; ++j) {
                double sum = 0;
                for (unsigned int k = 0; k < wA; ++k) {
                    double a = A[l * hA * wA + i * wA + k];
                    double b = B[l * wA * wB + k * wB + j];
                    sum += a * b;
                }
                C[l * hA * wB + i * wB + j] = (float)sum;
            }
}
template<unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D_bcbn(const float *input, const float *weight, float *output, int C, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);
    float accu[Batch/8][64] = {0};
    __shared__ float input_smem[Batch][bc][bn]; // TODO: 16 -> 100

    for(int i = 0; i < C; i+=bc) {
   
        //////// input transform //////
#pragma unroll
        for (int j = 0; j < Batch; j++) {
//            input_smem[j][threadIdx.x%bc][threadIdx.x/bc] = input[j * bc * bn + threadIdx.x];
            input_smem[j][warp_id][lane_id] = input[j * C * N + by * C * bn + lane_id * C + warp_id + i];
//            input_smem[j][warp_id][lane_id] = input[j * bc * bn + lane_id * bc + threadIdx.x];
        }
        //////////////////////////////
        __syncthreads();
    
        ////////////// load register ////////////
        float *ip = &input_smem[2*warp_id][0][0];
        const float *wp = &weight[2*warp_id*C*K+lane_id+i*K+bx*bk];
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
                accu[k][l] += ip[l-32] * wv;
              }
              wp += (K - 32); 
              ip += bn;
          }
          wp += (C - bc) * K;
        }
        __syncthreads();
    }
    unsigned int offset = by * bn * K + bx * bk;
#pragma unroll
    for (int i = 0; i < bn; i++) {
        output[offset + 2 * warp_id * N * K + i * K + lane_id] = accu[0][i];
        output[offset + 2 * warp_id * N * K + i * K + lane_id + 32] = accu[0][i+32];
        output[offset + (2 * warp_id + 1) * N * K + i * K + lane_id] = accu[1][i];
        output[offset + (2 * warp_id + 1) * N * K + i * K + lane_id + 32] = accu[1][i+32];
    }
}
template<unsigned int bn, unsigned int bc, unsigned int bk>
__global__ void winograd2D_bnbc(const float *input, const float *weight, float *output, int C, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int bx = blockIdx.x % (K / bk);
    int by = blockIdx.x / (K / bk);
    float accu[Batch/8][64] = {0};
    __shared__ float input_smem[Batch][bn][bc]; // TODO: 16 -> 100

    for(int i = 0; i < C; i+=bc) {
   
        //////// input transform //////
#pragma unroll
        for (int j = 0; j < Batch; j++) {
//            input_smem[j][threadIdx.x%bc][threadIdx.x/bc] = input[j * bc * bn + threadIdx.x];
            input_smem[j][lane_id][warp_id] = input[j * C * N + by * C * bn + lane_id * C + warp_id + i];
//            input_smem[j][warp_id][lane_id] = input[j * bc * bn + lane_id * bc + threadIdx.x];
        }
        //////////////////////////////
        __syncthreads();
    
        ////////////// load register ////////////
        float *ip = &input_smem[2*warp_id][0][0];
        const float *wp = &weight[2*warp_id*C*K+lane_id+i*K+bx*bk];
        /////// batched matmul 32x2x8 outer product//////////
#pragma unroll
        for(int k = 0; k < Batch/8; k++) {
#pragma unroll
          for(int j = 0; j < bc; j++) {
              float wv = wp[0];
              for(int l = 0; l < 32; l++) {
                accu[k][l] += input_smem[2*warp_id][l][j] * wv;
              }
              wp += 32; 
              wv = wp[0];
              for(int l = 32; l < 64; l++) {
                accu[k][l] += input_smem[2*warp_id][l-32][j] * wv;
              }
              wp += (K - 32); 
          }
          wp += (C - bc) * K;
        }
        __syncthreads();
    }
    unsigned int offset = by * bn * K + bx * bk;
#pragma unroll
    for (int i = 0; i < bn; i++) {
        output[offset + 2 * warp_id * N * K + i * K + lane_id] = accu[0][i];
        output[offset + 2 * warp_id * N * K + i * K + lane_id + 32] = accu[0][i+32];
        output[offset + (2 * warp_id + 1) * N * K + i * K + lane_id] = accu[1][i];
        output[offset + (2 * warp_id + 1) * N * K + i * K + lane_id + 32] = accu[1][i+32];
    }
}

template <typename T>
__global__ void forwardAssign2D(const T *Input, const T *Weight, T *tmp_data_buffer, const T **Input_ptrs_gpu, const T **Weight_ptrs_gpu, T **tmp_product_ptrs_gpu, int C, int B, int K) {
	int tx = threadIdx.x; // kernel_size
	
	Input_ptrs_gpu[tx] = Input + tx * B * C;
	Weight_ptrs_gpu[tx] = Weight + tx * K * C;
	tmp_product_ptrs_gpu[tx] = tmp_data_buffer + tx * B * K;
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
    unsigned int size_A = Batch * BN * BC;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = Batch * BC * BK;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
    float flop = 2 * (float)BN * (float)BC * (float)BK * Batch;
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
    unsigned int size_C = Batch * BN * BK;
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
    computeGold(reference, h_A, h_B, BN, BC, BK);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Naive CPU (Golden Reference)\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#endif

#if MY_bcbn == 1
    cudaDeviceSynchronize();
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
    winograd2D_bcbn<32, 8, 64><<<(BN/32)*(BK/64), 256>>>(d_A, d_B, d_C, BC, BN, BK);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Loop unrolling GPU 2\n");
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

#if MY_bnbc == 1
    cudaDeviceSynchronize();
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
    winograd2D_bnbc<32, 8, 64><<<(BN/32)*(BK/64), 256>>>(d_A, d_B, d_C, BC, BN, BK);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Loop unrolling GPU \n");
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

    /****************************************************/
    /*  Cublas                                          */
    /****************************************************/

#if CUBLAS == 1
    long long* tmp_ptr_buffer;
    cudaMalloc((void**) &tmp_ptr_buffer, 3*Batch*sizeof(long long));
    const float** Input_ptrs_gpu_ = (const float **)(tmp_ptr_buffer);
    const float** Weight_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + Batch);
    float** tmp_product_ptrs_gpu_ = (float **)(tmp_ptr_buffer + Batch * 2);

    dim3 bDim3(Batch, 1, 1);
    dim3 gDim3(1, 1, 1);
    forwardAssign2D <float> <<<gDim3, bDim3>>> (d_A, d_B, d_C, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, BC, BN, BK);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    float one = 1;
    float zero = 0;
  
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        BK, BN, BC,
        &one,
        Weight_ptrs_gpu_, BK,
        Input_ptrs_gpu_, BC,
        &zero, tmp_product_ptrs_gpu_, BK, Batch);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Cublas\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
    float* cublas_ref = (float*) malloc(mem_size_C);
    cudaMemcpy(cublas_ref, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
#endif

    // check result
#if CHECK_RESULT == 1
    printDiff(reference, h_C, BK, BN);
    free(reference);
#endif
#if CUBLAS == 1
    printDiff(cublas_ref, h_C, BK, BN);
    free(cublas_ref);
#endif
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k,l;
  int error_count=0;
    
  for (l=0; l<Batch; l++) {
    for (j=0; j<height; j++) {
      for (i=0; i<width; i++) {
        k = l*height*width+j*width+i;
        if (fabs(data1[k] - data2[k]) > 0.00001 ) {
           printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", i,j, data1[k], data2[k]);
           error_count++;
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
