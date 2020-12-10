#include <cublas_v2.h>

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (1, C, K)
// wino_weight = (4, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x1(const T *norm_weight, int C, int K, T *wino_weight)
{
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = norm_weight [0 * C * K + bx * K + tx];

    float product_weight_patch_0 = ele_0;
    float product_weight_patch_1 = ele_0;
    float product_weight_patch_2 = ele_0;
    float product_weight_patch_3 = ele_0;

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
	
}

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (2, C, K)
// wino_weight = (6, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x2(const T *norm_weight, int C, int K, T *wino_weight) 
{
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);

    float product_weight_patch_0 = T(ele_0);
    float product_weight_patch_1 = T(ele_0 + ele_1);
    float product_weight_patch_2 = T(ele_1);
    float product_weight_patch_3 = T(ele_0);
    float product_weight_patch_4 = T(ele_0 + ele_1);
    float product_weight_patch_5 = T(ele_1);

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
	
}

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (3, C, K)
// wino_weight = (8, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x3(const T *norm_weight, int C, int K, T *wino_weight) 
{ 
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);
    float ele_2 = float(norm_weight [2 * C * K + bx * K + tx]);

    float product_weight_patch_0 = ele_0;
    float product_weight_patch_1 = 1/2.*(ele_0 + ele_1 + ele_2);
    float product_weight_patch_2 = 1/2.*(ele_0 - ele_1 + ele_2);
    float product_weight_patch_3 = ele_2;
    float product_weight_patch_4 = ele_0;
    float product_weight_patch_5 = 1/2.*(ele_0 + ele_1 + ele_2);
    float product_weight_patch_6 = 1/2.*(ele_0 - ele_1 + ele_2);
    float product_weight_patch_7 = ele_2;


    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
    wino_weight [6 * C * K + bx * K + tx] = T(product_weight_patch_6);
    wino_weight [7 * C * K + bx * K + tx] = T(product_weight_patch_7);
	
}

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (2, C, K)
// wino_weight = (6, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x1(const T *norm_weight, int C, int K, T *wino_weight) 
{ 
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);

    float product_weight_patch_0 = ele_0;
    float product_weight_patch_1 = ele_0;
    float product_weight_patch_2 = ele_0 + ele_1;
    float product_weight_patch_3 = ele_0 + ele_1;
    float product_weight_patch_4 = ele_1;
    float product_weight_patch_5 = ele_1;

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
	
}

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (4, C, K)
// wino_weight = (9, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x2(const T *norm_weight, int C, int K, T *wino_weight) 
{ 
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);
    float ele_2 = float(norm_weight [2 * C * K + bx * K + tx]);
    float ele_3 = float(norm_weight [3 * C * K + bx * K + tx]);

    float product_weight_patch_0 = ele_0;
    float product_weight_patch_1 = ele_0 + ele_1;
    float product_weight_patch_2 = ele_1;
    float product_weight_patch_3 = ele_0 + ele_2;
    float product_weight_patch_4 = ele_0 + ele_1 + ele_2 + ele_3;
    float product_weight_patch_5 = ele_1 + ele_3;
    float product_weight_patch_6 = ele_2;
    float product_weight_patch_7 = ele_2 + ele_3;
    float product_weight_patch_8 = ele_3;

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
    wino_weight [6 * C * K + bx * K + tx] = T(product_weight_patch_6);
    wino_weight [7 * C * K + bx * K + tx] = T(product_weight_patch_7);
    wino_weight [8 * C * K + bx * K + tx] = T(product_weight_patch_8);
	
}

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (12, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x3(const T *norm_weight, int C, int K, T *wino_weight) 
{ 
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);
    float ele_2 = float(norm_weight [2 * C * K + bx * K + tx]);
    float ele_3 = float(norm_weight [3 * C * K + bx * K + tx]);
    float ele_4 = float(norm_weight [4 * C * K + bx * K + tx]);
    float ele_5 = float(norm_weight [5 * C * K + bx * K + tx]);

    float product_weight_patch_0 = ele_0;
    float product_weight_patch_1 = 1/2.*(ele_0 + ele_1 + ele_2);
    float product_weight_patch_2 = 1/2.*(ele_0 - ele_1 + ele_2);
    float product_weight_patch_3 = ele_2;
    float product_weight_patch_4 = ele_0 + ele_3;
    float product_weight_patch_5 = 1/2.*(ele_0 + ele_1 + ele_2 + ele_3 + ele_4 + ele_5);
    float product_weight_patch_6 = 1/2.*(ele_0 - ele_1 + ele_2 + ele_3 - ele_4 + ele_5);
    float product_weight_patch_7 = ele_2 + ele_5;
    float product_weight_patch_8 = ele_3;
    float product_weight_patch_9 = 1/2.*(ele_3 + ele_4 + ele_5);
    float product_weight_patch_10= 1/2.*(ele_3 - ele_4 + ele_5);
    float product_weight_patch_11= ele_5;

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
    wino_weight [6 * C * K + bx * K + tx] = T(product_weight_patch_6);
    wino_weight [7 * C * K + bx * K + tx] = T(product_weight_patch_7);
    wino_weight [8 * C * K + bx * K + tx] = T(product_weight_patch_8);
    wino_weight [9 * C * K + bx * K + tx] = T(product_weight_patch_9);
    wino_weight [10* C * K + bx * K + tx] = T(product_weight_patch_10);
    wino_weight [11* C * K + bx * K + tx] = T(product_weight_patch_11);
	
}

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (3, C, K)
// wino_weight = (8, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x1(const T *norm_weight, int C, int K, T *wino_weight) 
{ 
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);
    float ele_2 = float(norm_weight [2 * C * K + bx * K + tx]);

    float product_weight_patch_0 = ele_0;
    float product_weight_patch_1 = ele_0;
    float product_weight_patch_2 = 1/2.*(ele_0 + ele_1 + ele_2);
    float product_weight_patch_3 = 1/2.*(ele_0 + ele_1 + ele_2);
    float product_weight_patch_4 = 1/2.*(ele_0 - ele_1 + ele_2);
    float product_weight_patch_5 = 1/2.*(ele_0 - ele_1 + ele_2);
    float product_weight_patch_6 = ele_2;
    float product_weight_patch_7 = ele_2;

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
    wino_weight [6 * C * K + bx * K + tx] = T(product_weight_patch_6);
    wino_weight [7 * C * K + bx * K + tx] = T(product_weight_patch_7);
	
}

// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (12, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x2(const T *norm_weight, int C, int K, T *wino_weight) 
{ 
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);
    float ele_2 = float(norm_weight [2 * C * K + bx * K + tx]);
    float ele_3 = float(norm_weight [3 * C * K + bx * K + tx]);
    float ele_4 = float(norm_weight [4 * C * K + bx * K + tx]);
    float ele_5 = float(norm_weight [5 * C * K + bx * K + tx]);

    float product_weight_patch_0 = ele_0;
    float product_weight_patch_1 = ele_0 + ele_1;
    float product_weight_patch_2 = ele_1;
    float product_weight_patch_3 = 1/2.*(ele_0 + ele_2 + ele_4);
    float product_weight_patch_4 = 1/2.*(ele_0 + ele_1 + ele_2 + ele_3 + ele_4 + ele_5);
    float product_weight_patch_5 = 1/2.*(ele_1 + ele_3 + ele_5);
    float product_weight_patch_6 = 1/2.*(ele_0 - ele_2 + ele_4);
    float product_weight_patch_7 = 1/2.*(ele_0 + ele_1 - ele_2 - ele_3 + ele_4 + ele_5);
    float product_weight_patch_8 = 1/2.*(ele_1 - ele_3 + ele_5);
    float product_weight_patch_9 = ele_4;
    float product_weight_patch_10= ele_4 + ele_5;
    float product_weight_patch_11= ele_5;

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
    wino_weight [6 * C * K + bx * K + tx] = T(product_weight_patch_6);
    wino_weight [7 * C * K + bx * K + tx] = T(product_weight_patch_7);
    wino_weight [8 * C * K + bx * K + tx] = T(product_weight_patch_8);
    wino_weight [9 * C * K + bx * K + tx] = T(product_weight_patch_9);
    wino_weight [10* C * K + bx * K + tx] = T(product_weight_patch_10);
    wino_weight [11* C * K + bx * K + tx] = T(product_weight_patch_11);
	
}


// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (9, C, K)
// wino_weight = (16, C, K)

template <typename T>
__global__ void wNorm2WinoTransform3x3(const T *norm_weight, int C, int K, T *wino_weight) 
{ 
    int bx = blockIdx.x; // w
//    int by = blockIdx.y; // h
//    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // k

    float ele_0 = float(norm_weight [0 * C * K + bx * K + tx]);
    float ele_1 = float(norm_weight [1 * C * K + bx * K + tx]);
    float ele_2 = float(norm_weight [2 * C * K + bx * K + tx]);
    float ele_3 = float(norm_weight [3 * C * K + bx * K + tx]);
    float ele_4 = float(norm_weight [4 * C * K + bx * K + tx]);
    float ele_5 = float(norm_weight [5 * C * K + bx * K + tx]);
    float ele_6 = float(norm_weight [6 * C * K + bx * K + tx]);
    float ele_7 = float(norm_weight [7 * C * K + bx * K + tx]);
    float ele_8 = float(norm_weight [8 * C * K + bx * K + tx]);

    float product_weight_patch_0 = T(ele_0);
    float product_weight_patch_1 = T(1/2.*(ele_0 + ele_1 + ele_2));
    float product_weight_patch_2 = T(1/2.*(ele_0 - ele_1 + ele_2));
    float product_weight_patch_3 = T(ele_2);
    float product_weight_patch_4 = T(1/2.*(ele_0 + ele_3 + ele_6));
    float product_weight_patch_5 = T(1/4.*(ele_0 + ele_3 + ele_6 + ele_1 + ele_4 + ele_7 + ele_2 + ele_5 + ele_8));
    float product_weight_patch_6 = T(1/4.*(ele_0 + ele_3 + ele_6 - ele_1 - ele_4 - ele_7 + ele_2 + ele_5 + ele_8));
    float product_weight_patch_7 = T(1/2.*(ele_2 + ele_5 + ele_8));
    float product_weight_patch_8 = T(1/2.*(ele_0 - ele_3 + ele_6));
    float product_weight_patch_9 = T(1/4.*(ele_0 - ele_3 + ele_6 + ele_1 - ele_4 + ele_7 + ele_2 - ele_5 + ele_8));
    float product_weight_patch_10= T(1/4.*(ele_0 - ele_3 + ele_6 - ele_1 + ele_4 - ele_7 + ele_2 - ele_5 + ele_8));
    float product_weight_patch_11= T(1/2.*(ele_2 - ele_5 + ele_8));
    float product_weight_patch_12= T(ele_6);
    float product_weight_patch_13= T(1/2.*(ele_6 + ele_7 + ele_8));
    float product_weight_patch_14= T(1/2.*(ele_6 - ele_7 + ele_8));
    float product_weight_patch_15= T(ele_8);

    wino_weight [0 * C * K + bx * K + tx] = T(product_weight_patch_0);
    wino_weight [1 * C * K + bx * K + tx] = T(product_weight_patch_1);
    wino_weight [2 * C * K + bx * K + tx] = T(product_weight_patch_2);
    wino_weight [3 * C * K + bx * K + tx] = T(product_weight_patch_3);
    wino_weight [4 * C * K + bx * K + tx] = T(product_weight_patch_4);
    wino_weight [5 * C * K + bx * K + tx] = T(product_weight_patch_5);
    wino_weight [6 * C * K + bx * K + tx] = T(product_weight_patch_6);
    wino_weight [7 * C * K + bx * K + tx] = T(product_weight_patch_7);
    wino_weight [8 * C * K + bx * K + tx] = T(product_weight_patch_8);
    wino_weight [9 * C * K + bx * K + tx] = T(product_weight_patch_9);
    wino_weight [10* C * K + bx * K + tx] = T(product_weight_patch_10);
    wino_weight [11* C * K + bx * K + tx] = T(product_weight_patch_11);
    wino_weight [12* C * K + bx * K + tx] = T(product_weight_patch_12);
    wino_weight [13* C * K + bx * K + tx] = T(product_weight_patch_13);
    wino_weight [14* C * K + bx * K + tx] = T(product_weight_patch_14);
    wino_weight [15* C * K + bx * K + tx] = T(product_weight_patch_15);
	
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (4, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x1(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    
    // load (2, 3, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
  
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0;
    trans_input_patch_1 = input_patch_1;
    trans_input_patch_2 = input_patch_2;
    trans_input_patch_3 = input_patch_3;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (6, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x2(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    
    // load (2, 3, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 2; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0.0;
    f_x = xBase + 2; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0.0;
  
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_1;
    trans_input_patch_1 = input_patch_1;
    trans_input_patch_2 = input_patch_2 - input_patch_1;
    trans_input_patch_3 = input_patch_3 - input_patch_4;
    trans_input_patch_4 = input_patch_4;
    trans_input_patch_5 = input_patch_5 - input_patch_4;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (8, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x3(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    float input_patch_6;
    float input_patch_7;
    
    // load (2, 4, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 2; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 3; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0.0;
    f_x = xBase + 2; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_6 = 0.0;
    f_x = xBase + 3; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_7 = 0.0;
   
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    float trans_input_patch_6;
    float trans_input_patch_7;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_2;
    trans_input_patch_1 = input_patch_1 + input_patch_2;
    trans_input_patch_2 = input_patch_2 - input_patch_1;
    trans_input_patch_3 = input_patch_1 - input_patch_3;
    trans_input_patch_4 = input_patch_4 - input_patch_6;
    trans_input_patch_5 = input_patch_5 + input_patch_6;
    trans_input_patch_6 = input_patch_6 - input_patch_5;
    trans_input_patch_7 = input_patch_5 - input_patch_7;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
    wino_input [ 6 * stride + offset ] = T(trans_input_patch_6);
    wino_input [ 7 * stride + offset ] = T(trans_input_patch_7);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (6, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x1(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    
    // load (3, 2, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
    f_x = xBase + 0; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0.0;
    f_x = xBase + 1; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0.0;
    
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_2;
    trans_input_patch_1 = input_patch_1 - input_patch_3;
    trans_input_patch_2 = input_patch_2;
    trans_input_patch_3 = input_patch_3;
    trans_input_patch_4 = input_patch_4 - input_patch_2;
    trans_input_patch_5 = input_patch_5 - input_patch_3;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (9, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x2(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    float input_patch_6;
    float input_patch_7;
    float input_patch_8;
    
    // load (3, 3, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 2; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0.0;
    f_x = xBase + 2; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0.0;
    f_x = xBase + 0; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_6 = 0.0;
    f_x = xBase + 1; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_7 = 0.0;
    f_x = xBase + 2; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_8 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_8 = 0.0;
   
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    float trans_input_patch_6;
    float trans_input_patch_7;
    float trans_input_patch_8;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_1 - input_patch_3 + input_patch_4;
    trans_input_patch_1 = input_patch_1 - input_patch_4;
    trans_input_patch_2 = input_patch_2 - input_patch_1 + input_patch_4 - input_patch_5;
    trans_input_patch_3 = input_patch_3 - input_patch_4;
    trans_input_patch_4 = input_patch_4;
    trans_input_patch_5 = input_patch_5 - input_patch_4;
    trans_input_patch_6 = input_patch_4 - input_patch_3 + input_patch_6 - input_patch_7;
    trans_input_patch_7 = input_patch_7 - input_patch_4;
    trans_input_patch_8 = input_patch_4 - input_patch_5 - input_patch_7 + input_patch_8;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
    wino_input [ 6 * stride + offset ] = T(trans_input_patch_6);
    wino_input [ 7 * stride + offset ] = T(trans_input_patch_7);
    wino_input [ 8 * stride + offset ] = T(trans_input_patch_8);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (12, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x3(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    float input_patch_6;
    float input_patch_7;
    float input_patch_8;
    float input_patch_9;
    float input_patch_10;
    float input_patch_11;
    
    // load (3, 4, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 2; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 3; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0.0;
    f_x = xBase + 2; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_6 = 0.0;
    f_x = xBase + 3; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_7 = 0.0;
    f_x = xBase + 0; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_8 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_8 = 0.0;
    f_x = xBase + 1; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_9 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_9 = 0.0;
    f_x = xBase + 2; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_10 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_10 = 0.0;
    f_x = xBase + 3; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_11 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_11 = 0.0;
    
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    float trans_input_patch_6;
    float trans_input_patch_7;
    float trans_input_patch_8;
    float trans_input_patch_9;
    float trans_input_patch_10;
    float trans_input_patch_11;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_2 - input_patch_4 + input_patch_6;
    trans_input_patch_1 = input_patch_1 + input_patch_2 - input_patch_5 - input_patch_6;
    trans_input_patch_2 = input_patch_2 - input_patch_1 + input_patch_5 - input_patch_6;
    trans_input_patch_3 = input_patch_1 - input_patch_3 - input_patch_5 + input_patch_7;
    trans_input_patch_4 = input_patch_4 - input_patch_6;
    trans_input_patch_5 = input_patch_5 + input_patch_6;
    trans_input_patch_6 = input_patch_6 - input_patch_5;
    trans_input_patch_7 = input_patch_5 - input_patch_7;
    trans_input_patch_8 = input_patch_6 - input_patch_4 + input_patch_8 - input_patch_10;
    trans_input_patch_9 = input_patch_9 - input_patch_6 - input_patch_5 + input_patch_10;
    trans_input_patch_10 = input_patch_5 - input_patch_6 - input_patch_9 + input_patch_10;
    trans_input_patch_11 = input_patch_7 - input_patch_5 + input_patch_9 - input_patch_11;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
    wino_input [ 6 * stride + offset ] = T(trans_input_patch_6);
    wino_input [ 7 * stride + offset ] = T(trans_input_patch_7);
    wino_input [ 8 * stride + offset ] = T(trans_input_patch_8);
    wino_input [ 9 * stride + offset ] = T(trans_input_patch_9);
    wino_input [ 10* stride + offset ] = T(trans_input_patch_10);
    wino_input [ 11* stride + offset ] = T(trans_input_patch_11);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (8, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x1(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    float input_patch_6;
    float input_patch_7;
    
    // load (4, 2, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
    f_x = xBase + 0; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0.0;
    f_x = xBase + 1; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0.0;
    f_x = xBase + 0; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_6 = 0.0;
    f_x = xBase + 1; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_7 = 0.0;
    
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    float trans_input_patch_6;
    float trans_input_patch_7;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_4;
    trans_input_patch_1 = input_patch_1 - input_patch_5;
    trans_input_patch_2 = input_patch_2 + input_patch_4;
    trans_input_patch_3 = input_patch_3 + input_patch_5;
    trans_input_patch_4 = input_patch_4 - input_patch_2;
    trans_input_patch_5 = input_patch_5 - input_patch_3;
    trans_input_patch_6 = input_patch_2 - input_patch_6;
    trans_input_patch_7 = input_patch_3 - input_patch_7;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
    wino_input [ 6 * stride + offset ] = T(trans_input_patch_6);
    wino_input [ 7 * stride + offset ] = T(trans_input_patch_7);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (12, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x2(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    float input_patch_6;
    float input_patch_7;
    float input_patch_8;
    float input_patch_9;
    float input_patch_10;
    float input_patch_11;
    
    // load (4, 3, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0.0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0.0;
    f_x = xBase + 2; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0.0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0.0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0.0;
    f_x = xBase + 2; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0.0;
    f_x = xBase + 0; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_6 = 0.0;
    f_x = xBase + 1; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_7 = 0.0;
    f_x = xBase + 2; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_8 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_8 = 0.0;
    f_x = xBase + 0; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_9 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_9 = 0.0;
    f_x = xBase + 1; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_10 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_10 = 0.0;
    f_x = xBase + 2; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_11 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_11 = 0.0;
    
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    float trans_input_patch_6;
    float trans_input_patch_7;
    float trans_input_patch_8;
    float trans_input_patch_9;
    float trans_input_patch_10;
    float trans_input_patch_11;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_1 - input_patch_6 + input_patch_7;
    trans_input_patch_1 = input_patch_1 - input_patch_7;
    trans_input_patch_2 = input_patch_2 - input_patch_1 + input_patch_7 - input_patch_8;
    trans_input_patch_3 = input_patch_3 - input_patch_4 + input_patch_6 - input_patch_7;
    trans_input_patch_4 = input_patch_4 + input_patch_7;
    trans_input_patch_5 = input_patch_5 - input_patch_4 - input_patch_7 + input_patch_8;
    trans_input_patch_6 = input_patch_4 - input_patch_3 + input_patch_6 - input_patch_7;
    trans_input_patch_7 = input_patch_7 - input_patch_4;
    trans_input_patch_8 = input_patch_4 - input_patch_5 - input_patch_7 + input_patch_8;
    trans_input_patch_9 = input_patch_3 - input_patch_4 - input_patch_9 + input_patch_10;
    trans_input_patch_10= input_patch_4 - input_patch_10;
    trans_input_patch_11= input_patch_5 - input_patch_4 + input_patch_10 - input_patch_11;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
    wino_input [ 6 * stride + offset ] = T(trans_input_patch_6);
    wino_input [ 7 * stride + offset ] = T(trans_input_patch_7);
    wino_input [ 8 * stride + offset ] = T(trans_input_patch_8);
    wino_input [ 9 * stride + offset ] = T(trans_input_patch_9);
    wino_input [ 10* stride + offset ] = T(trans_input_patch_10);
    wino_input [ 11* stride + offset ] = T(trans_input_patch_11);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (16, Batch, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x3(const T *norm_input, int C, int B, int H, int W, int pad_h, int pad_w, T *wino_input)
{ 
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int t = threadIdx.x; // c
    
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    
    int f_b = bz;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * by - pad_h;
    
    // float input_patch_1 [16] = {0};
    float input_patch_0;
    float input_patch_1;
    float input_patch_2;
    float input_patch_3;
    float input_patch_4;
    float input_patch_5;
    float input_patch_6;
    float input_patch_7;
    float input_patch_8;
    float input_patch_9;
    float input_patch_10;
    float input_patch_11;
    float input_patch_12;
    float input_patch_13;
    float input_patch_14;
    float input_patch_15;
    
    // load (4, 4, 1) patch of input from global memory
    int f_x, f_y;
    f_x = xBase + 0; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_0 = 0;
    f_x = xBase + 1; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_1 = 0;
    f_x = xBase + 2; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_2 = 0;
    f_x = xBase + 3; f_y = yBase + 0;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_3 = 0;
    f_x = xBase + 0; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_4 = 0;
    f_x = xBase + 1; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_5 = 0;
    f_x = xBase + 2; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_6 = 0;
    f_x = xBase + 3; f_y = yBase + 1;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_7 = 0;
    f_x = xBase + 0; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_8 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_8 = 0;
    f_x = xBase + 1; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_9 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_9 = 0;
    f_x = xBase + 2; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_10 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_10 = 0;
    f_x = xBase + 3; f_y = yBase + 2;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_11 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_11 = 0;
    f_x = xBase + 0; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_12 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_12 = 0;
    f_x = xBase + 1; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_13 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_13 = 0;
    f_x = xBase + 2; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_14 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_14 = 0;
    f_x = xBase + 3; f_y = yBase + 3;
    if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_15 = float(norm_input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]); 
    else input_patch_15 = 0;
    
    float trans_input_patch_0;
    float trans_input_patch_1;
    float trans_input_patch_2;
    float trans_input_patch_3;
    float trans_input_patch_4;
    float trans_input_patch_5;
    float trans_input_patch_6;
    float trans_input_patch_7;
    float trans_input_patch_8;
    float trans_input_patch_9;
    float trans_input_patch_10;
    float trans_input_patch_11;
    float trans_input_patch_12;
    float trans_input_patch_13;
    float trans_input_patch_14;
    float trans_input_patch_15;
    
    // Winograd Transform
    trans_input_patch_0 = input_patch_0 - input_patch_2 - input_patch_8 + input_patch_10;
    trans_input_patch_1 = input_patch_1 + input_patch_2 - input_patch_9 - input_patch_10;
    trans_input_patch_2 = input_patch_2 - input_patch_1 + input_patch_9 - input_patch_10;
    trans_input_patch_3 = input_patch_1 - input_patch_3 - input_patch_9 + input_patch_11;
    trans_input_patch_4 = input_patch_4 - input_patch_6 + input_patch_8 - input_patch_10;
    trans_input_patch_5 = input_patch_5 + input_patch_6 + input_patch_9 + input_patch_10;
    trans_input_patch_6 = input_patch_6 - input_patch_5 - input_patch_9 + input_patch_10;
    trans_input_patch_7 = input_patch_5 - input_patch_7 + input_patch_9 - input_patch_11;
    trans_input_patch_8 = input_patch_6 - input_patch_4 + input_patch_8 - input_patch_10;
    trans_input_patch_9 = input_patch_9 - input_patch_6 - input_patch_5 + input_patch_10;
    trans_input_patch_10= input_patch_5 - input_patch_6 - input_patch_9 + input_patch_10;
    trans_input_patch_11= input_patch_7 - input_patch_5 + input_patch_9 - input_patch_11;
    trans_input_patch_12= input_patch_4 - input_patch_6 - input_patch_12 + input_patch_14;
    trans_input_patch_13= input_patch_5 + input_patch_6 - input_patch_13 - input_patch_14;
    trans_input_patch_14= input_patch_6 - input_patch_5 + input_patch_13 - input_patch_14;
    trans_input_patch_15= input_patch_5 - input_patch_7 - input_patch_13 + input_patch_15;
    
    
    int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
    int stride = B * nH * nW * C;
    
    wino_input [ 0 * stride + offset ] = T(trans_input_patch_0);
    wino_input [ 1 * stride + offset ] = T(trans_input_patch_1);
    wino_input [ 2 * stride + offset ] = T(trans_input_patch_2);
    wino_input [ 3 * stride + offset ] = T(trans_input_patch_3);
    wino_input [ 4 * stride + offset ] = T(trans_input_patch_4);
    wino_input [ 5 * stride + offset ] = T(trans_input_patch_5);
    wino_input [ 6 * stride + offset ] = T(trans_input_patch_6);
    wino_input [ 7 * stride + offset ] = T(trans_input_patch_7);
    wino_input [ 8 * stride + offset ] = T(trans_input_patch_8);
    wino_input [ 9 * stride + offset ] = T(trans_input_patch_9);
    wino_input [ 10* stride + offset ] = T(trans_input_patch_10);
    wino_input [ 11* stride + offset ] = T(trans_input_patch_11);
    wino_input [ 12* stride + offset ] = T(trans_input_patch_12);
    wino_input [ 13* stride + offset ] = T(trans_input_patch_13);
    wino_input [ 14* stride + offset ] = T(trans_input_patch_14);
    wino_input [ 15* stride + offset ] = T(trans_input_patch_15);
} 

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (4, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x1(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0;
    float output_patch_1 = product_patch_1;
    float output_patch_2 = product_patch_2;
    float output_patch_3 = product_patch_3;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (6, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x2(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_1;
    float output_patch_1 = product_patch_1 + product_patch_2;
    float output_patch_2 = product_patch_3 + product_patch_4;
    float output_patch_3 = product_patch_4 + product_patch_5;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (8, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x3(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_6 = float(wino_output [6 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_7 = float(wino_output [7 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_1 + product_patch_2;
    float output_patch_1 = product_patch_1 - product_patch_2 - product_patch_3;
    float output_patch_2 = product_patch_4 + product_patch_5 + product_patch_6;
    float output_patch_3 = product_patch_5 - product_patch_6 - product_patch_7;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (6, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x1(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_2;
    float output_patch_1 = product_patch_1 + product_patch_3;
    float output_patch_2 = product_patch_2 + product_patch_4;
    float output_patch_3 = product_patch_3 + product_patch_5;

    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
												
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (9, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x2(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_6 = float(wino_output [6 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_7 = float(wino_output [7 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_8 = float(wino_output [8 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_1 + product_patch_3 + product_patch_4;
    float output_patch_1 = product_patch_1 + product_patch_2 + product_patch_4 + product_patch_5;
    float output_patch_2 = product_patch_3 + product_patch_4 + product_patch_6 + product_patch_7;
    float output_patch_3 = product_patch_4 + product_patch_5 + product_patch_7 + product_patch_8;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
												
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (12, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x3(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_6 = float(wino_output [6 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_7 = float(wino_output [7 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_8 = float(wino_output [8 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_9 = float(wino_output [9 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_10= float(wino_output [10* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_11= float(wino_output [11* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_1 + product_patch_2 + product_patch_4 + product_patch_5 + product_patch_6;
    float output_patch_1 = product_patch_1 - product_patch_2 - product_patch_3 + product_patch_5 - product_patch_6 - product_patch_7;
    float output_patch_2 = product_patch_4 + product_patch_5 + product_patch_6 + product_patch_8 + product_patch_9 + product_patch_10;
    float output_patch_3 = product_patch_5 - product_patch_6 - product_patch_7 + product_patch_9 - product_patch_10- product_patch_11;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (8, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x1(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_6 = float(wino_output [6 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_7 = float(wino_output [7 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_2 + product_patch_4;
    float output_patch_1 = product_patch_1 + product_patch_3 + product_patch_5;
    float output_patch_2 = product_patch_2 - product_patch_4 - product_patch_6;
    float output_patch_3 = product_patch_3 - product_patch_5 - product_patch_7;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (12, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x2(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_6 = float(wino_output [6 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_7 = float(wino_output [7 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_8 = float(wino_output [8 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_9 = float(wino_output [9 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_10= float(wino_output [10* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_11= float(wino_output [11* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_1 + product_patch_3 + product_patch_4 + product_patch_6 + product_patch_7;
    float output_patch_1 = product_patch_1 + product_patch_2 + product_patch_4 + product_patch_5 + product_patch_7 + product_patch_8;
    float output_patch_2 = product_patch_3 + product_patch_4 - product_patch_6 - product_patch_7 - product_patch_9 - product_patch_10;
    float output_patch_3 = product_patch_4 + product_patch_5 - product_patch_7 - product_patch_8 - product_patch_10- product_patch_11;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// wino_output = (16, Batch, nH, nW, K)
// norm_output = (Batch, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x3(const T *wino_output, int B, int output_H, int output_W, int K, T *norm_output)
{
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b 
    int tx = threadIdx.x; // K
    int nH;
    int nW;
    nH = (output_H+1)/2;
    nW = (output_W+1)/2;

    float product_patch_0 = float(wino_output [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_1 = float(wino_output [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_2 = float(wino_output [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_3 = float(wino_output [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_4 = float(wino_output [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_5 = float(wino_output [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_6 = float(wino_output [6 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_7 = float(wino_output [7 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_8 = float(wino_output [8 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_9 = float(wino_output [9 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_10= float(wino_output [10* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_11= float(wino_output [11* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_12= float(wino_output [12* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_13= float(wino_output [13* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_14= float(wino_output [14* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
    float product_patch_15= float(wino_output [15* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx]);
	
    float output_patch_0 = product_patch_0 + product_patch_1 + product_patch_2 + product_patch_4 +
                       product_patch_5 + product_patch_6 + product_patch_8 + product_patch_9 + product_patch_10;
    float output_patch_1 = product_patch_1 - product_patch_2 - product_patch_3 + product_patch_5 -
                       product_patch_6 - product_patch_7 + product_patch_9 - product_patch_10 - product_patch_11;
    float output_patch_2 = product_patch_4 + product_patch_5 + product_patch_6 - product_patch_8 -
                       product_patch_9 - product_patch_10 - product_patch_12 - product_patch_13 - product_patch_14;
    float output_patch_3 = product_patch_5 - product_patch_6 - product_patch_7 - product_patch_9 +
                       product_patch_10 + product_patch_11 - product_patch_13 + product_patch_14 + product_patch_15;
    
    norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_0);
    if(output_W%2==0 || bx!=nW-1)
        norm_output[bz*output_H*output_W*K + (2*by+0)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_1);
    if(output_H%2==0 || by!=nH-1)
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+0)*K + tx] = T(output_patch_2);
    if((output_W%2==0 || bx!=nW-1) && (output_H%2==0 || by!=nH-1))
        norm_output[bz*output_H*output_W*K + (2*by+1)*output_W*K + (2*bx+1)*K + tx] = T(output_patch_3);
} 

