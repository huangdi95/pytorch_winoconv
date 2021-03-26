/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Tue 23 Mar 2021 10:41:51 AM CST
 ************************************************************************/
#include<iostream>
#include<stdio.h>
using namespace std;

__global__ void lds128(float *data, float *a) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    __shared__ float smem[256]; // TODO: 16 -> 100
    smem[4*threadIdx.x] = data[threadIdx.x];
//    smem[4*threadIdx.x+1] = data[4*threadIdx.x+1];
//    smem[4*threadIdx.x+2] = data[4*threadIdx.x+2];
//    smem[4*threadIdx.x+3] = data[4*threadIdx.x+3];
//    smem[threadIdx.x] = data[threadIdx.x];
//    smem[threadIdx.x+64] = data[threadIdx.x+64];
//    smem[threadIdx.x+128] = data[threadIdx.x+128];
//    smem[threadIdx.x+192] = data[threadIdx.x+192];
    __syncthreads();
    float c[266] = {300};
    for(int i = 0; i < 164; i++) {
        a[i] = c[i];
    }
//    ((float4*)a)[threadIdx.x] = ((float4*)smem)[threadIdx.x];
//    smem[4*threadIdx.x] = c[0];
//    smem[4*threadIdx.x+1] = c[threadIdx.x+1];
//    smem[4*threadIdx.x+2] = c[threadIdx.x+2];
//    smem[4*threadIdx.x+3] = c[threadIdx.x+3];
    ((float4*)a)[threadIdx.x] = ((float4*)smem)[threadIdx.x];
    __syncthreads();
///    float b[8] = {0};
///    float *ip = &smem[0];
///    b[0] += ip[0]*a[0];
///    b[1] += ip[1]*a[1];
///    b[2] += ip[2]*a[2];
///    b[3] += ip[3]*a[3];
///    b[4] += ip[0]*a[4];
///    b[5] += ip[1]*a[5];
///    b[6] += ip[2]*a[6];
///    b[7] += ip[3]*a[7];
///    ((float4*)a)[0] = ((float4*)b)[0];
///    ((float4*)a)[1] = ((float4*)b)[1];
//    ((float4*)b)[0] = ((float4*)smem)[0];
//    float b[2][3][4];
//    ((float4*)b[0][0])[0] = ((float4*)smem)[threadIdx.x];
//    ((float4*)b[1][0])[0] = ((float4*)smem)[threadIdx.x];
//    ((float4*)a)[threadIdx.x] = ((float4*)b[0][0])[0];
//    ((float4*)a)[threadIdx.x] = ((float4*)smem)[threadIdx.x];
//    a[threadIdx.x] = smem[threadIdx.x];
//    a[1] = ((float*)smem)[threadIdx.x+1];
//    a[2] = ((float*)smem)[threadIdx.x+2];
//    a[3] = ((float*)smem)[threadIdx.x+3];
//    b[0] = ((float4*)smem)[threadIdx.x];
//    ((float4*)smem)[threadIdx.x] = b[0];

}

int main() {
    float data_host[256]; 
    for (int i = 0; i < 256; i++)
        data_host[i] = i;
    
    float *data_dev = nullptr;
    cudaMalloc((void**)&data_dev, 256*sizeof(float));
    cudaMemcpy(data_dev, data_host, 256*sizeof(float), cudaMemcpyHostToDevice);
    float *a_dev = nullptr;
    cudaMalloc((void**)&a_dev, 256*sizeof(float));
    float *b_dev = nullptr;
    cudaMalloc((void**)&b_dev, 256*sizeof(float));
    lds128<<<1, 64>>>(data_dev, a_dev);
    float a_host[256];
    cudaMemcpy(a_host, a_dev, 256*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; i++) {
        cout << data_host[i] << " ";
    }
    cout << endl;
    for (int i = 0; i < 256; i++) {
        cout << a_host[i] << " ";
    }
    cout << endl;

    return 0;
}
