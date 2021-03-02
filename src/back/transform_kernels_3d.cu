

template <typename T>
__global__ void wNorm2WinoTransform1x1x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[1] = {T(0)};
    for(int i = 0; i < 1; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[8] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = ele[0];
    product_weight_patch[3] = ele[0];
    product_weight_patch[4] = ele[0];
    product_weight_patch[5] = ele[0];
    product_weight_patch[6] = ele[0];
    product_weight_patch[7] = ele[0];


    for(int i = 0; i < 8; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}
// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (2, C, K)
// wino_weight = (12, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x1x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[2] = {T(0)};
    for(int i = 0; i < 2; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[12] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = ele[0];
    product_weight_patch[4] = ele[0] + ele[1];
    product_weight_patch[5] = ele[1];
    product_weight_patch[6] = ele[0];
    product_weight_patch[7] = ele[0] + ele[1];
    product_weight_patch[8] = ele[1];
    product_weight_patch[9] = ele[0];
    product_weight_patch[10] = ele[0] + ele[1];
    product_weight_patch[11] = ele[1];


    for(int i = 0; i < 12; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (12, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x1x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[8] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[4 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[4 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[8] = {T(0)};


    trans_input_patch[0] = input_patch[0];
    trans_input_patch[1] = input_patch[1];
    trans_input_patch[2] = input_patch[2];
    trans_input_patch[3] = input_patch[3];
    trans_input_patch[4] = input_patch[4];
    trans_input_patch[5] = input_patch[5];
    trans_input_patch[6] = input_patch[6];
    trans_input_patch[7] = input_patch[7];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 8; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}
// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (12, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x1x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[12] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[6 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[6 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[12] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[1];
    trans_input_patch[1] = input_patch[1];
    trans_input_patch[2] = -input_patch[1] + input_patch[2];
    trans_input_patch[3] = input_patch[3] - input_patch[4];
    trans_input_patch[4] = input_patch[4];
    trans_input_patch[5] = -input_patch[4] + input_patch[5];
    trans_input_patch[6] = input_patch[6] - input_patch[7];
    trans_input_patch[7] = input_patch[7];
    trans_input_patch[8] = -input_patch[7] + input_patch[8];
    trans_input_patch[9] = -input_patch[10] + input_patch[9];
    trans_input_patch[10] = input_patch[10];
    trans_input_patch[11] = -input_patch[10] + input_patch[11];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 12; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (12, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x1x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[8] = {T(0)};

    for(int i = 0; i < 8; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0];
    output_patch[1] = product_patch[1];
    output_patch[2] = product_patch[2];
    output_patch[3] = product_patch[3];
    output_patch[4] = product_patch[4];
    output_patch[5] = product_patch[5];
    output_patch[6] = product_patch[6];
    output_patch[7] = product_patch[7];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}


// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (12, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x1x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[12] = {T(0)};

    for(int i = 0; i < 12; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[1];
    output_patch[1] = product_patch[1] + product_patch[2];
    output_patch[2] = product_patch[3] + product_patch[4];
    output_patch[3] = product_patch[4] + product_patch[5];
    output_patch[4] = product_patch[6] + product_patch[7];
    output_patch[5] = product_patch[7] + product_patch[8];
    output_patch[6] = product_patch[10] + product_patch[9];
    output_patch[7] = product_patch[10] + product_patch[11];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (3, C, K)
// wino_weight = (16, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x1x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[3] = {T(0)};
    for(int i = 0; i < 3; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[16] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = ele[0];
    product_weight_patch[5] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[7] = ele[2];
    product_weight_patch[8] = ele[0];
    product_weight_patch[9] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[10] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[11] = ele[2];
    product_weight_patch[12] = ele[0];
    product_weight_patch[13] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[14] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[15] = ele[2];


    for(int i = 0; i < 16; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (16, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x1x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[16] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[8 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[8 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[16] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[2];
    trans_input_patch[1] = input_patch[1] + input_patch[2];
    trans_input_patch[2] = -input_patch[1] + input_patch[2];
    trans_input_patch[3] = input_patch[1] - input_patch[3];
    trans_input_patch[4] = input_patch[4] - input_patch[6];
    trans_input_patch[5] = input_patch[5] + input_patch[6];
    trans_input_patch[6] = -input_patch[5] + input_patch[6];
    trans_input_patch[7] = input_patch[5] - input_patch[7];
    trans_input_patch[8] = -input_patch[10] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] + input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14];
    trans_input_patch[13] = input_patch[13] + input_patch[14];
    trans_input_patch[14] = -input_patch[13] + input_patch[14];
    trans_input_patch[15] = input_patch[13] - input_patch[15];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 16; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (16, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x1x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[16] = {T(0)};

    for(int i = 0; i < 16; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[1] + product_patch[2];
    output_patch[1] = product_patch[1] - product_patch[2] - product_patch[3];
    output_patch[2] = product_patch[4] + product_patch[5] + product_patch[6];
    output_patch[3] = product_patch[5] - product_patch[6] - product_patch[7];
    output_patch[4] = product_patch[10] + product_patch[8] + product_patch[9];
    output_patch[5] = -product_patch[10] - product_patch[11] + product_patch[9];
    output_patch[6] = product_patch[12] + product_patch[13] + product_patch[14];
    output_patch[7] = product_patch[13] - product_patch[14] - product_patch[15];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (2, C, K)
// wino_weight = (12, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x2x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[2] = {T(0)};
    for(int i = 0; i < 2; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[12] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = ele[0] + ele[1];
    product_weight_patch[3] = ele[0] + ele[1];
    product_weight_patch[4] = ele[1];
    product_weight_patch[5] = ele[1];
    product_weight_patch[6] = ele[0];
    product_weight_patch[7] = ele[0];
    product_weight_patch[8] = ele[0] + ele[1];
    product_weight_patch[9] = ele[0] + ele[1];
    product_weight_patch[10] = ele[1];
    product_weight_patch[11] = ele[1];


    for(int i = 0; i < 12; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (12, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x2x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[12] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[6 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[6 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[12] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[2];
    trans_input_patch[1] = input_patch[1] - input_patch[3];
    trans_input_patch[2] = input_patch[2];
    trans_input_patch[3] = input_patch[3];
    trans_input_patch[4] = -input_patch[2] + input_patch[4];
    trans_input_patch[5] = -input_patch[3] + input_patch[5];
    trans_input_patch[6] = input_patch[6] - input_patch[8];
    trans_input_patch[7] = input_patch[7] - input_patch[9];
    trans_input_patch[8] = input_patch[8];
    trans_input_patch[9] = input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[8];
    trans_input_patch[11] = input_patch[11] - input_patch[9];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 12; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (12, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x2x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[12] = {T(0)};

    for(int i = 0; i < 12; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[2];
    output_patch[1] = product_patch[1] + product_patch[3];
    output_patch[2] = product_patch[2] + product_patch[4];
    output_patch[3] = product_patch[3] + product_patch[5];
    output_patch[4] = product_patch[6] + product_patch[8];
    output_patch[5] = product_patch[7] + product_patch[9];
    output_patch[6] = product_patch[10] + product_patch[8];
    output_patch[7] = product_patch[11] + product_patch[9];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (4, C, K)
// wino_weight = (18, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x2x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[4] = {T(0)};
    for(int i = 0; i < 4; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[18] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = ele[0] + ele[2];
    product_weight_patch[4] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[5] = ele[1] + ele[3];
    product_weight_patch[6] = ele[2];
    product_weight_patch[7] = ele[2] + ele[3];
    product_weight_patch[8] = ele[3];
    product_weight_patch[9] = ele[0];
    product_weight_patch[10] = ele[0] + ele[1];
    product_weight_patch[11] = ele[1];
    product_weight_patch[12] = ele[0] + ele[2];
    product_weight_patch[13] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[14] = ele[1] + ele[3];
    product_weight_patch[15] = ele[2];
    product_weight_patch[16] = ele[2] + ele[3];
    product_weight_patch[17] = ele[3];


    for(int i = 0; i < 18; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (18, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x2x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[18] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[9 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[9 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[18] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[1] - input_patch[3] + input_patch[4];
    trans_input_patch[1] = input_patch[1] - input_patch[4];
    trans_input_patch[2] = -input_patch[1] + input_patch[2] + input_patch[4] - input_patch[5];
    trans_input_patch[3] = input_patch[3] - input_patch[4];
    trans_input_patch[4] = input_patch[4];
    trans_input_patch[5] = -input_patch[4] + input_patch[5];
    trans_input_patch[6] = -input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[7] = -input_patch[4] + input_patch[7];
    trans_input_patch[8] = input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[9] = -input_patch[10] - input_patch[12] + input_patch[13] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[13];
    trans_input_patch[11] = -input_patch[10] + input_patch[11] + input_patch[13] - input_patch[14];
    trans_input_patch[12] = input_patch[12] - input_patch[13];
    trans_input_patch[13] = input_patch[13];
    trans_input_patch[14] = -input_patch[13] + input_patch[14];
    trans_input_patch[15] = -input_patch[12] + input_patch[13] + input_patch[15] - input_patch[16];
    trans_input_patch[16] = -input_patch[13] + input_patch[16];
    trans_input_patch[17] = input_patch[13] - input_patch[14] - input_patch[16] + input_patch[17];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 18; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (18, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x2x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[18] = {T(0)};

    for(int i = 0; i < 18; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[1] + product_patch[3] + product_patch[4];
    output_patch[1] = product_patch[1] + product_patch[2] + product_patch[4] + product_patch[5];
    output_patch[2] = product_patch[3] + product_patch[4] + product_patch[6] + product_patch[7];
    output_patch[3] = product_patch[4] + product_patch[5] + product_patch[7] + product_patch[8];
    output_patch[4] = product_patch[10] + product_patch[12] + product_patch[13] + product_patch[9];
    output_patch[5] = product_patch[10] + product_patch[11] + product_patch[13] + product_patch[14];
    output_patch[6] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16];
    output_patch[7] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (24, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x2x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[6] = {T(0)};
    for(int i = 0; i < 6; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[24] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = ele[0] + ele[3];
    product_weight_patch[5] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[7] = ele[2] + ele[5];
    product_weight_patch[8] = ele[3];
    product_weight_patch[9] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[10] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[11] = ele[5];
    product_weight_patch[12] = ele[0];
    product_weight_patch[13] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[14] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[15] = ele[2];
    product_weight_patch[16] = ele[0] + ele[3];
    product_weight_patch[17] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[18] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[19] = ele[2] + ele[5];
    product_weight_patch[20] = ele[3];
    product_weight_patch[21] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[22] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[23] = ele[5];


    for(int i = 0; i < 24; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (24, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x2x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[24] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[12 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[12 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[24] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[2] - input_patch[4] + input_patch[6];
    trans_input_patch[1] = input_patch[1] + input_patch[2] - input_patch[5] - input_patch[6];
    trans_input_patch[2] = -input_patch[1] + input_patch[2] + input_patch[5] - input_patch[6];
    trans_input_patch[3] = input_patch[1] - input_patch[3] - input_patch[5] + input_patch[7];
    trans_input_patch[4] = input_patch[4] - input_patch[6];
    trans_input_patch[5] = input_patch[5] + input_patch[6];
    trans_input_patch[6] = -input_patch[5] + input_patch[6];
    trans_input_patch[7] = input_patch[5] - input_patch[7];
    trans_input_patch[8] = -input_patch[10] - input_patch[4] + input_patch[6] + input_patch[8];
    trans_input_patch[9] = input_patch[10] - input_patch[5] - input_patch[6] + input_patch[9];
    trans_input_patch[10] = input_patch[10] + input_patch[5] - input_patch[6] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] - input_patch[5] + input_patch[7] + input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14] - input_patch[16] + input_patch[18];
    trans_input_patch[13] = input_patch[13] + input_patch[14] - input_patch[17] - input_patch[18];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[17] - input_patch[18];
    trans_input_patch[15] = input_patch[13] - input_patch[15] - input_patch[17] + input_patch[19];
    trans_input_patch[16] = input_patch[16] - input_patch[18];
    trans_input_patch[17] = input_patch[17] + input_patch[18];
    trans_input_patch[18] = -input_patch[17] + input_patch[18];
    trans_input_patch[19] = input_patch[17] - input_patch[19];
    trans_input_patch[20] = -input_patch[16] + input_patch[18] + input_patch[20] - input_patch[22];
    trans_input_patch[21] = -input_patch[17] - input_patch[18] + input_patch[21] + input_patch[22];
    trans_input_patch[22] = input_patch[17] - input_patch[18] - input_patch[21] + input_patch[22];
    trans_input_patch[23] = -input_patch[17] + input_patch[19] + input_patch[21] - input_patch[23];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 24; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (24, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x2x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[24] = {T(0)};

    for(int i = 0; i < 24; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[1] + product_patch[2] + product_patch[4] + product_patch[5] + product_patch[6];
    output_patch[1] = product_patch[1] - product_patch[2] - product_patch[3] + product_patch[5] - product_patch[6] - product_patch[7];
    output_patch[2] = product_patch[10] + product_patch[4] + product_patch[5] + product_patch[6] + product_patch[8] + product_patch[9];
    output_patch[3] = -product_patch[10] - product_patch[11] + product_patch[5] - product_patch[6] - product_patch[7] + product_patch[9];
    output_patch[4] = product_patch[12] + product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[18];
    output_patch[5] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[17] - product_patch[18] - product_patch[19];
    output_patch[6] = product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22];
    output_patch[7] = product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (3, C, K)
// wino_weight = (16, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x3x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[3] = {T(0)};
    for(int i = 0; i < 3; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[16] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[3] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[4] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[5] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[6] = ele[2];
    product_weight_patch[7] = ele[2];
    product_weight_patch[8] = ele[0];
    product_weight_patch[9] = ele[0];
    product_weight_patch[10] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[11] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[12] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[13] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[14] = ele[2];
    product_weight_patch[15] = ele[2];


    for(int i = 0; i < 16; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (16, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x3x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[16] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[8 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[8 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[16] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[4];
    trans_input_patch[1] = input_patch[1] - input_patch[5];
    trans_input_patch[2] = input_patch[2] + input_patch[4];
    trans_input_patch[3] = input_patch[3] + input_patch[5];
    trans_input_patch[4] = -input_patch[2] + input_patch[4];
    trans_input_patch[5] = -input_patch[3] + input_patch[5];
    trans_input_patch[6] = input_patch[2] - input_patch[6];
    trans_input_patch[7] = input_patch[3] - input_patch[7];
    trans_input_patch[8] = -input_patch[12] + input_patch[8];
    trans_input_patch[9] = -input_patch[13] + input_patch[9];
    trans_input_patch[10] = input_patch[10] + input_patch[12];
    trans_input_patch[11] = input_patch[11] + input_patch[13];
    trans_input_patch[12] = -input_patch[10] + input_patch[12];
    trans_input_patch[13] = -input_patch[11] + input_patch[13];
    trans_input_patch[14] = input_patch[10] - input_patch[14];
    trans_input_patch[15] = input_patch[11] - input_patch[15];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 16; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (16, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x3x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[16] = {T(0)};

    for(int i = 0; i < 16; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[2] + product_patch[4];
    output_patch[1] = product_patch[1] + product_patch[3] + product_patch[5];
    output_patch[2] = product_patch[2] - product_patch[4] - product_patch[6];
    output_patch[3] = product_patch[3] - product_patch[5] - product_patch[7];
    output_patch[4] = product_patch[10] + product_patch[12] + product_patch[8];
    output_patch[5] = product_patch[11] + product_patch[13] + product_patch[9];
    output_patch[6] = product_patch[10] - product_patch[12] - product_patch[14];
    output_patch[7] = product_patch[11] - product_patch[13] - product_patch[15];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (24, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x3x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[6] = {T(0)};
    for(int i = 0; i < 6; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[24] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[4] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[5] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[7] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[8] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[9] = ele[4];
    product_weight_patch[10] = ele[4] + ele[5];
    product_weight_patch[11] = ele[5];
    product_weight_patch[12] = ele[0];
    product_weight_patch[13] = ele[0] + ele[1];
    product_weight_patch[14] = ele[1];
    product_weight_patch[15] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[16] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[17] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[18] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[19] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[20] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[21] = ele[4];
    product_weight_patch[22] = ele[4] + ele[5];
    product_weight_patch[23] = ele[5];


    for(int i = 0; i < 24; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (24, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x3x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[24] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[12 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[12 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[24] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[1] - input_patch[6] + input_patch[7];
    trans_input_patch[1] = input_patch[1] - input_patch[7];
    trans_input_patch[2] = -input_patch[1] + input_patch[2] + input_patch[7] - input_patch[8];
    trans_input_patch[3] = input_patch[3] - input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[4] = input_patch[4] + input_patch[7];
    trans_input_patch[5] = -input_patch[4] + input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[6] = -input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[7] = -input_patch[4] + input_patch[7];
    trans_input_patch[8] = input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[3] - input_patch[4] - input_patch[9];
    trans_input_patch[10] = -input_patch[10] + input_patch[4];
    trans_input_patch[11] = input_patch[10] - input_patch[11] - input_patch[4] + input_patch[5];
    trans_input_patch[12] = input_patch[12] - input_patch[13] - input_patch[18] + input_patch[19];
    trans_input_patch[13] = input_patch[13] - input_patch[19];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[19] - input_patch[20];
    trans_input_patch[15] = input_patch[15] - input_patch[16] + input_patch[18] - input_patch[19];
    trans_input_patch[16] = input_patch[16] + input_patch[19];
    trans_input_patch[17] = -input_patch[16] + input_patch[17] - input_patch[19] + input_patch[20];
    trans_input_patch[18] = -input_patch[15] + input_patch[16] + input_patch[18] - input_patch[19];
    trans_input_patch[19] = -input_patch[16] + input_patch[19];
    trans_input_patch[20] = input_patch[16] - input_patch[17] - input_patch[19] + input_patch[20];
    trans_input_patch[21] = input_patch[15] - input_patch[16] - input_patch[21] + input_patch[22];
    trans_input_patch[22] = input_patch[16] - input_patch[22];
    trans_input_patch[23] = -input_patch[16] + input_patch[17] + input_patch[22] - input_patch[23];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 24; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (24, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x3x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[24] = {T(0)};

    for(int i = 0; i < 24; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[1] + product_patch[3] + product_patch[4] + product_patch[6] + product_patch[7];
    output_patch[1] = product_patch[1] + product_patch[2] + product_patch[4] + product_patch[5] + product_patch[7] + product_patch[8];
    output_patch[2] = -product_patch[10] + product_patch[3] + product_patch[4] - product_patch[6] - product_patch[7] - product_patch[9];
    output_patch[3] = -product_patch[10] - product_patch[11] + product_patch[4] + product_patch[5] - product_patch[7] - product_patch[8];
    output_patch[4] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[18] + product_patch[19];
    output_patch[5] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[19] + product_patch[20];
    output_patch[6] = product_patch[15] + product_patch[16] - product_patch[18] - product_patch[19] - product_patch[21] - product_patch[22];
    output_patch[7] = product_patch[16] + product_patch[17] - product_patch[19] - product_patch[20] - product_patch[22] - product_patch[23];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (9, C, K)
// wino_weight = (32, C, K)
template <typename T>
__global__ void wNorm2WinoTransform1x3x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[9] = {T(0)};
    for(int i = 0; i < 9; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[32] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[5] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[6] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[7] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[8] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[9] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[10] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[11] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[12] = ele[6];
    product_weight_patch[13] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[14] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[15] = ele[8];
    product_weight_patch[16] = ele[0];
    product_weight_patch[17] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[18] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[19] = ele[2];
    product_weight_patch[20] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[21] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[22] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[23] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[24] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[25] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[26] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[27] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[28] = ele[6];
    product_weight_patch[29] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[30] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[31] = ele[8];


    for(int i = 0; i < 32; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (32, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform1x3x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 2) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[32] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[16 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[16 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[32] = {T(0)};


    trans_input_patch[0] = input_patch[0] + input_patch[10] - input_patch[2] - input_patch[8];
    trans_input_patch[1] = -input_patch[10] + input_patch[1] + input_patch[2] - input_patch[9];
    trans_input_patch[2] = -input_patch[10] - input_patch[1] + input_patch[2] + input_patch[9];
    trans_input_patch[3] = input_patch[11] + input_patch[1] - input_patch[3] - input_patch[9];
    trans_input_patch[4] = -input_patch[10] + input_patch[4] - input_patch[6] + input_patch[8];
    trans_input_patch[5] = input_patch[10] + input_patch[5] + input_patch[6] + input_patch[9];
    trans_input_patch[6] = input_patch[10] - input_patch[5] + input_patch[6] - input_patch[9];
    trans_input_patch[7] = -input_patch[11] + input_patch[5] - input_patch[7] + input_patch[9];
    trans_input_patch[8] = -input_patch[10] - input_patch[4] + input_patch[6] + input_patch[8];
    trans_input_patch[9] = input_patch[10] - input_patch[5] - input_patch[6] + input_patch[9];
    trans_input_patch[10] = input_patch[10] + input_patch[5] - input_patch[6] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] - input_patch[5] + input_patch[7] + input_patch[9];
    trans_input_patch[12] = -input_patch[12] + input_patch[14] + input_patch[4] - input_patch[6];
    trans_input_patch[13] = -input_patch[13] - input_patch[14] + input_patch[5] + input_patch[6];
    trans_input_patch[14] = input_patch[13] - input_patch[14] - input_patch[5] + input_patch[6];
    trans_input_patch[15] = -input_patch[13] + input_patch[15] + input_patch[5] - input_patch[7];
    trans_input_patch[16] = input_patch[16] - input_patch[18] - input_patch[24] + input_patch[26];
    trans_input_patch[17] = input_patch[17] + input_patch[18] - input_patch[25] - input_patch[26];
    trans_input_patch[18] = -input_patch[17] + input_patch[18] + input_patch[25] - input_patch[26];
    trans_input_patch[19] = input_patch[17] - input_patch[19] - input_patch[25] + input_patch[27];
    trans_input_patch[20] = input_patch[20] - input_patch[22] + input_patch[24] - input_patch[26];
    trans_input_patch[21] = input_patch[21] + input_patch[22] + input_patch[25] + input_patch[26];
    trans_input_patch[22] = -input_patch[21] + input_patch[22] - input_patch[25] + input_patch[26];
    trans_input_patch[23] = input_patch[21] - input_patch[23] + input_patch[25] - input_patch[27];
    trans_input_patch[24] = -input_patch[20] + input_patch[22] + input_patch[24] - input_patch[26];
    trans_input_patch[25] = -input_patch[21] - input_patch[22] + input_patch[25] + input_patch[26];
    trans_input_patch[26] = input_patch[21] - input_patch[22] - input_patch[25] + input_patch[26];
    trans_input_patch[27] = -input_patch[21] + input_patch[23] + input_patch[25] - input_patch[27];
    trans_input_patch[28] = input_patch[20] - input_patch[22] - input_patch[28] + input_patch[30];
    trans_input_patch[29] = input_patch[21] + input_patch[22] - input_patch[29] - input_patch[30];
    trans_input_patch[30] = -input_patch[21] + input_patch[22] + input_patch[29] - input_patch[30];
    trans_input_patch[31] = input_patch[21] - input_patch[23] - input_patch[29] + input_patch[31];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 32; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (32, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform1x3x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[32] = {T(0)};

    for(int i = 0; i < 32; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[1] + product_patch[2] + product_patch[4] + product_patch[5] + product_patch[6] + product_patch[8] + product_patch[9];
    output_patch[1] = -product_patch[10] - product_patch[11] + product_patch[1] - product_patch[2] - product_patch[3] + product_patch[5] - product_patch[6] - product_patch[7] + product_patch[9];
    output_patch[2] = -product_patch[10] - product_patch[12] - product_patch[13] - product_patch[14] + product_patch[4] + product_patch[5] + product_patch[6] - product_patch[8] - product_patch[9];
    output_patch[3] = product_patch[10] + product_patch[11] - product_patch[13] + product_patch[14] + product_patch[15] + product_patch[5] - product_patch[6] - product_patch[7] - product_patch[9];
    output_patch[4] = product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[24] + product_patch[25] + product_patch[26];
    output_patch[5] = product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[25] - product_patch[26] - product_patch[27];
    output_patch[6] = product_patch[20] + product_patch[21] + product_patch[22] - product_patch[24] - product_patch[25] - product_patch[26] - product_patch[28] - product_patch[29] - product_patch[30];
    output_patch[7] = product_patch[21] - product_patch[22] - product_patch[23] - product_patch[25] + product_patch[26] + product_patch[27] - product_patch[29] + product_patch[30] + product_patch[31];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (2, C, K)
// wino_weight = (12, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x1x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[2] = {T(0)};
    for(int i = 0; i < 2; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[12] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = ele[0];
    product_weight_patch[3] = ele[0];
    product_weight_patch[4] = ele[0] + ele[1];
    product_weight_patch[5] = ele[0] + ele[1];
    product_weight_patch[6] = ele[0] + ele[1];
    product_weight_patch[7] = ele[0] + ele[1];
    product_weight_patch[8] = ele[1];
    product_weight_patch[9] = ele[1];
    product_weight_patch[10] = ele[1];
    product_weight_patch[11] = ele[1];


    for(int i = 0; i < 12; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (12, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x1x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[12] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[4 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[4 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[12] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[4];
    trans_input_patch[1] = input_patch[1] - input_patch[5];
    trans_input_patch[2] = input_patch[2] - input_patch[6];
    trans_input_patch[3] = input_patch[3] - input_patch[7];
    trans_input_patch[4] = input_patch[4];
    trans_input_patch[5] = input_patch[5];
    trans_input_patch[6] = input_patch[6];
    trans_input_patch[7] = input_patch[7];
    trans_input_patch[8] = -input_patch[4] + input_patch[8];
    trans_input_patch[9] = -input_patch[5] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[6];
    trans_input_patch[11] = input_patch[11] - input_patch[7];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 12; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (12, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x1x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[12] = {T(0)};

    for(int i = 0; i < 12; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[4];
    output_patch[1] = product_patch[1] + product_patch[5];
    output_patch[2] = product_patch[2] + product_patch[6];
    output_patch[3] = product_patch[3] + product_patch[7];
    output_patch[4] = product_patch[4] + product_patch[8];
    output_patch[5] = product_patch[5] + product_patch[9];
    output_patch[6] = product_patch[10] + product_patch[6];
    output_patch[7] = product_patch[11] + product_patch[7];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (4, C, K)
// wino_weight = (18, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x1x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[4] = {T(0)};
    for(int i = 0; i < 4; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[18] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = ele[0];
    product_weight_patch[4] = ele[0] + ele[1];
    product_weight_patch[5] = ele[1];
    product_weight_patch[6] = ele[0] + ele[2];
    product_weight_patch[7] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[8] = ele[1] + ele[3];
    product_weight_patch[9] = ele[0] + ele[2];
    product_weight_patch[10] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[11] = ele[1] + ele[3];
    product_weight_patch[12] = ele[2];
    product_weight_patch[13] = ele[2] + ele[3];
    product_weight_patch[14] = ele[3];
    product_weight_patch[15] = ele[2];
    product_weight_patch[16] = ele[2] + ele[3];
    product_weight_patch[17] = ele[3];


    for(int i = 0; i < 18; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (18, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x1x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[18] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[6 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[6 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[18] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[1] - input_patch[6] + input_patch[7];
    trans_input_patch[1] = input_patch[1] - input_patch[7];
    trans_input_patch[2] = -input_patch[1] + input_patch[2] + input_patch[7] - input_patch[8];
    trans_input_patch[3] = input_patch[10] + input_patch[3] - input_patch[4] - input_patch[9];
    trans_input_patch[4] = -input_patch[10] + input_patch[4];
    trans_input_patch[5] = input_patch[10] - input_patch[11] - input_patch[4] + input_patch[5];
    trans_input_patch[6] = input_patch[6] - input_patch[7];
    trans_input_patch[7] = input_patch[7];
    trans_input_patch[8] = -input_patch[7] + input_patch[8];
    trans_input_patch[9] = -input_patch[10] + input_patch[9];
    trans_input_patch[10] = input_patch[10];
    trans_input_patch[11] = -input_patch[10] + input_patch[11];
    trans_input_patch[12] = input_patch[12] - input_patch[13] - input_patch[6] + input_patch[7];
    trans_input_patch[13] = input_patch[13] - input_patch[7];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[7] - input_patch[8];
    trans_input_patch[15] = input_patch[10] + input_patch[15] - input_patch[16] - input_patch[9];
    trans_input_patch[16] = -input_patch[10] + input_patch[16];
    trans_input_patch[17] = input_patch[10] - input_patch[11] - input_patch[16] + input_patch[17];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 18; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (18, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x1x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[18] = {T(0)};

    for(int i = 0; i < 18; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[1] + product_patch[6] + product_patch[7];
    output_patch[1] = product_patch[1] + product_patch[2] + product_patch[7] + product_patch[8];
    output_patch[2] = product_patch[10] + product_patch[3] + product_patch[4] + product_patch[9];
    output_patch[3] = product_patch[10] + product_patch[11] + product_patch[4] + product_patch[5];
    output_patch[4] = product_patch[12] + product_patch[13] + product_patch[6] + product_patch[7];
    output_patch[5] = product_patch[13] + product_patch[14] + product_patch[7] + product_patch[8];
    output_patch[6] = product_patch[10] + product_patch[15] + product_patch[16] + product_patch[9];
    output_patch[7] = product_patch[10] + product_patch[11] + product_patch[16] + product_patch[17];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (24, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x1x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[6] = {T(0)};
    for(int i = 0; i < 6; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[24] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = ele[0];
    product_weight_patch[5] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[7] = ele[2];
    product_weight_patch[8] = ele[0] + ele[3];
    product_weight_patch[9] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[10] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[11] = ele[2] + ele[5];
    product_weight_patch[12] = ele[0] + ele[3];
    product_weight_patch[13] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[14] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[15] = ele[2] + ele[5];
    product_weight_patch[16] = ele[3];
    product_weight_patch[17] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[18] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[19] = ele[5];
    product_weight_patch[20] = ele[3];
    product_weight_patch[21] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[22] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[23] = ele[5];


    for(int i = 0; i < 24; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (24, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x1x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[24] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[8 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[8 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[24] = {T(0)};


    trans_input_patch[0] = input_patch[0] + input_patch[10] - input_patch[2] - input_patch[8];
    trans_input_patch[1] = -input_patch[10] + input_patch[1] + input_patch[2] - input_patch[9];
    trans_input_patch[2] = -input_patch[10] - input_patch[1] + input_patch[2] + input_patch[9];
    trans_input_patch[3] = input_patch[11] + input_patch[1] - input_patch[3] - input_patch[9];
    trans_input_patch[4] = -input_patch[12] + input_patch[14] + input_patch[4] - input_patch[6];
    trans_input_patch[5] = -input_patch[13] - input_patch[14] + input_patch[5] + input_patch[6];
    trans_input_patch[6] = input_patch[13] - input_patch[14] - input_patch[5] + input_patch[6];
    trans_input_patch[7] = -input_patch[13] + input_patch[15] + input_patch[5] - input_patch[7];
    trans_input_patch[8] = -input_patch[10] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] + input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14];
    trans_input_patch[13] = input_patch[13] + input_patch[14];
    trans_input_patch[14] = -input_patch[13] + input_patch[14];
    trans_input_patch[15] = input_patch[13] - input_patch[15];
    trans_input_patch[16] = input_patch[10] + input_patch[16] - input_patch[18] - input_patch[8];
    trans_input_patch[17] = -input_patch[10] + input_patch[17] + input_patch[18] - input_patch[9];
    trans_input_patch[18] = -input_patch[10] - input_patch[17] + input_patch[18] + input_patch[9];
    trans_input_patch[19] = input_patch[11] + input_patch[17] - input_patch[19] - input_patch[9];
    trans_input_patch[20] = -input_patch[12] + input_patch[14] + input_patch[20] - input_patch[22];
    trans_input_patch[21] = -input_patch[13] - input_patch[14] + input_patch[21] + input_patch[22];
    trans_input_patch[22] = input_patch[13] - input_patch[14] - input_patch[21] + input_patch[22];
    trans_input_patch[23] = -input_patch[13] + input_patch[15] + input_patch[21] - input_patch[23];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 24; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (24, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x1x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[24] = {T(0)};

    for(int i = 0; i < 24; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[1] + product_patch[2] + product_patch[8] + product_patch[9];
    output_patch[1] = -product_patch[10] - product_patch[11] + product_patch[1] - product_patch[2] - product_patch[3] + product_patch[9];
    output_patch[2] = product_patch[12] + product_patch[13] + product_patch[14] + product_patch[4] + product_patch[5] + product_patch[6];
    output_patch[3] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[5] - product_patch[6] - product_patch[7];
    output_patch[4] = product_patch[10] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[8] + product_patch[9];
    output_patch[5] = -product_patch[10] - product_patch[11] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[9];
    output_patch[6] = product_patch[12] + product_patch[13] + product_patch[14] + product_patch[20] + product_patch[21] + product_patch[22];
    output_patch[7] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[21] - product_patch[22] - product_patch[23];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (4, C, K)
// wino_weight = (18, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x2x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[4] = {T(0)};
    for(int i = 0; i < 4; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[18] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = ele[0] + ele[1];
    product_weight_patch[3] = ele[0] + ele[1];
    product_weight_patch[4] = ele[1];
    product_weight_patch[5] = ele[1];
    product_weight_patch[6] = ele[0] + ele[2];
    product_weight_patch[7] = ele[0] + ele[2];
    product_weight_patch[8] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[9] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[10] = ele[1] + ele[3];
    product_weight_patch[11] = ele[1] + ele[3];
    product_weight_patch[12] = ele[2];
    product_weight_patch[13] = ele[2];
    product_weight_patch[14] = ele[2] + ele[3];
    product_weight_patch[15] = ele[2] + ele[3];
    product_weight_patch[16] = ele[3];
    product_weight_patch[17] = ele[3];


    for(int i = 0; i < 18; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (18, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x2x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[18] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[6 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[6 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[18] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[2] - input_patch[6] + input_patch[8];
    trans_input_patch[1] = input_patch[1] - input_patch[3] - input_patch[7] + input_patch[9];
    trans_input_patch[2] = input_patch[2] - input_patch[8];
    trans_input_patch[3] = input_patch[3] - input_patch[9];
    trans_input_patch[4] = -input_patch[10] - input_patch[2] + input_patch[4] + input_patch[8];
    trans_input_patch[5] = -input_patch[11] - input_patch[3] + input_patch[5] + input_patch[9];
    trans_input_patch[6] = input_patch[6] - input_patch[8];
    trans_input_patch[7] = input_patch[7] - input_patch[9];
    trans_input_patch[8] = input_patch[8];
    trans_input_patch[9] = input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[8];
    trans_input_patch[11] = input_patch[11] - input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14] - input_patch[6] + input_patch[8];
    trans_input_patch[13] = input_patch[13] - input_patch[15] - input_patch[7] + input_patch[9];
    trans_input_patch[14] = input_patch[14] - input_patch[8];
    trans_input_patch[15] = input_patch[15] - input_patch[9];
    trans_input_patch[16] = -input_patch[10] - input_patch[14] + input_patch[16] + input_patch[8];
    trans_input_patch[17] = -input_patch[11] - input_patch[15] + input_patch[17] + input_patch[9];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 18; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (18, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x2x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[18] = {T(0)};

    for(int i = 0; i < 18; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[2] + product_patch[6] + product_patch[8];
    output_patch[1] = product_patch[1] + product_patch[3] + product_patch[7] + product_patch[9];
    output_patch[2] = product_patch[10] + product_patch[2] + product_patch[4] + product_patch[8];
    output_patch[3] = product_patch[11] + product_patch[3] + product_patch[5] + product_patch[9];
    output_patch[4] = product_patch[12] + product_patch[14] + product_patch[6] + product_patch[8];
    output_patch[5] = product_patch[13] + product_patch[15] + product_patch[7] + product_patch[9];
    output_patch[6] = product_patch[10] + product_patch[14] + product_patch[16] + product_patch[8];
    output_patch[7] = product_patch[11] + product_patch[15] + product_patch[17] + product_patch[9];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (8, C, K)
// wino_weight = (27, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x2x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[8] = {T(0)};
    for(int i = 0; i < 8; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[27] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = ele[0] + ele[2];
    product_weight_patch[4] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[5] = ele[1] + ele[3];
    product_weight_patch[6] = ele[2];
    product_weight_patch[7] = ele[2] + ele[3];
    product_weight_patch[8] = ele[3];
    product_weight_patch[9] = ele[0] + ele[4];
    product_weight_patch[10] = ele[0] + ele[1] + ele[4] + ele[5];
    product_weight_patch[11] = ele[1] + ele[5];
    product_weight_patch[12] = ele[0] + ele[2] + ele[4] + ele[6];
    product_weight_patch[13] = ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7];
    product_weight_patch[14] = ele[1] + ele[3] + ele[5] + ele[7];
    product_weight_patch[15] = ele[2] + ele[6];
    product_weight_patch[16] = ele[2] + ele[3] + ele[6] + ele[7];
    product_weight_patch[17] = ele[3] + ele[7];
    product_weight_patch[18] = ele[4];
    product_weight_patch[19] = ele[4] + ele[5];
    product_weight_patch[20] = ele[5];
    product_weight_patch[21] = ele[4] + ele[6];
    product_weight_patch[22] = ele[4] + ele[5] + ele[6] + ele[7];
    product_weight_patch[23] = ele[5] + ele[7];
    product_weight_patch[24] = ele[6];
    product_weight_patch[25] = ele[6] + ele[7];
    product_weight_patch[26] = ele[7];


    for(int i = 0; i < 27; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (27, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x2x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[27] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[9 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[9 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[27] = {T(0)};


    trans_input_patch[0] = input_patch[0] + input_patch[10] + input_patch[12] - input_patch[13] - input_patch[1] - input_patch[3] + input_patch[4] - input_patch[9];
    trans_input_patch[1] = -input_patch[10] + input_patch[13] + input_patch[1] - input_patch[4];
    trans_input_patch[2] = input_patch[10] - input_patch[11] - input_patch[13] + input_patch[14] - input_patch[1] + input_patch[2] + input_patch[4] - input_patch[5];
    trans_input_patch[3] = -input_patch[12] + input_patch[13] + input_patch[3] - input_patch[4];
    trans_input_patch[4] = -input_patch[13] + input_patch[4];
    trans_input_patch[5] = input_patch[13] - input_patch[14] - input_patch[4] + input_patch[5];
    trans_input_patch[6] = input_patch[12] - input_patch[13] - input_patch[15] + input_patch[16] - input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[7] = input_patch[13] - input_patch[16] - input_patch[4] + input_patch[7];
    trans_input_patch[8] = -input_patch[13] + input_patch[14] + input_patch[16] - input_patch[17] + input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[9] = -input_patch[10] - input_patch[12] + input_patch[13] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[13];
    trans_input_patch[11] = -input_patch[10] + input_patch[11] + input_patch[13] - input_patch[14];
    trans_input_patch[12] = input_patch[12] - input_patch[13];
    trans_input_patch[13] = input_patch[13];
    trans_input_patch[14] = -input_patch[13] + input_patch[14];
    trans_input_patch[15] = -input_patch[12] + input_patch[13] + input_patch[15] - input_patch[16];
    trans_input_patch[16] = -input_patch[13] + input_patch[16];
    trans_input_patch[17] = input_patch[13] - input_patch[14] - input_patch[16] + input_patch[17];
    trans_input_patch[18] = input_patch[10] + input_patch[12] - input_patch[13] + input_patch[18] - input_patch[19] - input_patch[21] + input_patch[22] - input_patch[9];
    trans_input_patch[19] = -input_patch[10] + input_patch[13] + input_patch[19] - input_patch[22];
    trans_input_patch[20] = input_patch[10] - input_patch[11] - input_patch[13] + input_patch[14] - input_patch[19] + input_patch[20] + input_patch[22] - input_patch[23];
    trans_input_patch[21] = -input_patch[12] + input_patch[13] + input_patch[21] - input_patch[22];
    trans_input_patch[22] = -input_patch[13] + input_patch[22];
    trans_input_patch[23] = input_patch[13] - input_patch[14] - input_patch[22] + input_patch[23];
    trans_input_patch[24] = input_patch[12] - input_patch[13] - input_patch[15] + input_patch[16] - input_patch[21] + input_patch[22] + input_patch[24] - input_patch[25];
    trans_input_patch[25] = input_patch[13] - input_patch[16] - input_patch[22] + input_patch[25];
    trans_input_patch[26] = -input_patch[13] + input_patch[14] + input_patch[16] - input_patch[17] + input_patch[22] - input_patch[23] - input_patch[25] + input_patch[26];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 27; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (27, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x2x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[27] = {T(0)};

    for(int i = 0; i < 27; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[12] + product_patch[13] + product_patch[1] + product_patch[3] + product_patch[4] + product_patch[9];
    output_patch[1] = product_patch[10] + product_patch[11] + product_patch[13] + product_patch[14] + product_patch[1] + product_patch[2] + product_patch[4] + product_patch[5];
    output_patch[2] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[3] + product_patch[4] + product_patch[6] + product_patch[7];
    output_patch[3] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[4] + product_patch[5] + product_patch[7] + product_patch[8];
    output_patch[4] = product_patch[10] + product_patch[12] + product_patch[13] + product_patch[18] + product_patch[19] + product_patch[21] + product_patch[22] + product_patch[9];
    output_patch[5] = product_patch[10] + product_patch[11] + product_patch[13] + product_patch[14] + product_patch[19] + product_patch[20] + product_patch[22] + product_patch[23];
    output_patch[6] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[21] + product_patch[22] + product_patch[24] + product_patch[25];
    output_patch[7] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[22] + product_patch[23] + product_patch[25] + product_patch[26];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (12, C, K)
// wino_weight = (36, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x2x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[12] = {T(0)};
    for(int i = 0; i < 12; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[36] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = ele[0] + ele[3];
    product_weight_patch[5] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[7] = ele[2] + ele[5];
    product_weight_patch[8] = ele[3];
    product_weight_patch[9] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[10] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[11] = ele[5];
    product_weight_patch[12] = ele[0] + ele[6];
    product_weight_patch[13] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[14] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[15] = ele[2] + ele[8];
    product_weight_patch[16] = ele[0] + ele[3] + ele[6] + ele[9];
    product_weight_patch[17] = T(1/2.)*(ele[0] + ele[10] + ele[11] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[18] = T(1/2.)*(ele[0] - ele[10] + ele[11] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[19] = ele[11] + ele[2] + ele[5] + ele[8];
    product_weight_patch[20] = ele[3] + ele[9];
    product_weight_patch[21] = T(1/2.)*(ele[10] + ele[11] + ele[3] + ele[4] + ele[5] + ele[9]);
    product_weight_patch[22] = T(1/2.)*(-ele[10] + ele[11] + ele[3] - ele[4] + ele[5] + ele[9]);
    product_weight_patch[23] = ele[11] + ele[5];
    product_weight_patch[24] = ele[6];
    product_weight_patch[25] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[26] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[27] = ele[8];
    product_weight_patch[28] = ele[6] + ele[9];
    product_weight_patch[29] = T(1/2.)*(ele[10] + ele[11] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[30] = T(1/2.)*(-ele[10] + ele[11] + ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[31] = ele[11] + ele[8];
    product_weight_patch[32] = ele[9];
    product_weight_patch[33] = T(1/2.)*(ele[10] + ele[11] + ele[9]);
    product_weight_patch[34] = T(1/2.)*(-ele[10] + ele[11] + ele[9]);
    product_weight_patch[35] = ele[11];


    for(int i = 0; i < 36; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (36, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x2x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[36] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[12 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[12 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[36] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[12] + input_patch[14] + input_patch[16] - input_patch[18] - input_patch[2] - input_patch[4] + input_patch[6];
    trans_input_patch[1] = -input_patch[13] - input_patch[14] + input_patch[17] + input_patch[18] + input_patch[1] + input_patch[2] - input_patch[5] - input_patch[6];
    trans_input_patch[2] = input_patch[13] - input_patch[14] - input_patch[17] + input_patch[18] - input_patch[1] + input_patch[2] + input_patch[5] - input_patch[6];
    trans_input_patch[3] = -input_patch[13] + input_patch[15] + input_patch[17] - input_patch[19] + input_patch[1] - input_patch[3] - input_patch[5] + input_patch[7];
    trans_input_patch[4] = -input_patch[16] + input_patch[18] + input_patch[4] - input_patch[6];
    trans_input_patch[5] = -input_patch[17] - input_patch[18] + input_patch[5] + input_patch[6];
    trans_input_patch[6] = input_patch[17] - input_patch[18] - input_patch[5] + input_patch[6];
    trans_input_patch[7] = -input_patch[17] + input_patch[19] + input_patch[5] - input_patch[7];
    trans_input_patch[8] = -input_patch[10] + input_patch[16] - input_patch[18] - input_patch[20] + input_patch[22] - input_patch[4] + input_patch[6] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[17] + input_patch[18] - input_patch[21] - input_patch[22] - input_patch[5] - input_patch[6] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[17] + input_patch[18] + input_patch[21] - input_patch[22] + input_patch[5] - input_patch[6] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] + input_patch[17] - input_patch[19] - input_patch[21] + input_patch[23] - input_patch[5] + input_patch[7] + input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14] - input_patch[16] + input_patch[18];
    trans_input_patch[13] = input_patch[13] + input_patch[14] - input_patch[17] - input_patch[18];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[17] - input_patch[18];
    trans_input_patch[15] = input_patch[13] - input_patch[15] - input_patch[17] + input_patch[19];
    trans_input_patch[16] = input_patch[16] - input_patch[18];
    trans_input_patch[17] = input_patch[17] + input_patch[18];
    trans_input_patch[18] = -input_patch[17] + input_patch[18];
    trans_input_patch[19] = input_patch[17] - input_patch[19];
    trans_input_patch[20] = -input_patch[16] + input_patch[18] + input_patch[20] - input_patch[22];
    trans_input_patch[21] = -input_patch[17] - input_patch[18] + input_patch[21] + input_patch[22];
    trans_input_patch[22] = input_patch[17] - input_patch[18] - input_patch[21] + input_patch[22];
    trans_input_patch[23] = -input_patch[17] + input_patch[19] + input_patch[21] - input_patch[23];
    trans_input_patch[24] = -input_patch[12] + input_patch[14] + input_patch[16] - input_patch[18] + input_patch[24] - input_patch[26] - input_patch[28] + input_patch[30];
    trans_input_patch[25] = -input_patch[13] - input_patch[14] + input_patch[17] + input_patch[18] + input_patch[25] + input_patch[26] - input_patch[29] - input_patch[30];
    trans_input_patch[26] = input_patch[13] - input_patch[14] - input_patch[17] + input_patch[18] - input_patch[25] + input_patch[26] + input_patch[29] - input_patch[30];
    trans_input_patch[27] = -input_patch[13] + input_patch[15] + input_patch[17] - input_patch[19] + input_patch[25] - input_patch[27] - input_patch[29] + input_patch[31];
    trans_input_patch[28] = -input_patch[16] + input_patch[18] + input_patch[28] - input_patch[30];
    trans_input_patch[29] = -input_patch[17] - input_patch[18] + input_patch[29] + input_patch[30];
    trans_input_patch[30] = input_patch[17] - input_patch[18] - input_patch[29] + input_patch[30];
    trans_input_patch[31] = -input_patch[17] + input_patch[19] + input_patch[29] - input_patch[31];
    trans_input_patch[32] = input_patch[16] - input_patch[18] - input_patch[20] + input_patch[22] - input_patch[28] + input_patch[30] + input_patch[32] - input_patch[34];
    trans_input_patch[33] = input_patch[17] + input_patch[18] - input_patch[21] - input_patch[22] - input_patch[29] - input_patch[30] + input_patch[33] + input_patch[34];
    trans_input_patch[34] = -input_patch[17] + input_patch[18] + input_patch[21] - input_patch[22] + input_patch[29] - input_patch[30] - input_patch[33] + input_patch[34];
    trans_input_patch[35] = input_patch[17] - input_patch[19] - input_patch[21] + input_patch[23] - input_patch[29] + input_patch[31] + input_patch[33] - input_patch[35];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 36; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (36, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x2x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[36] = {T(0)};

    for(int i = 0; i < 36; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[12] + product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[1] + product_patch[2] + product_patch[4] + product_patch[5] + product_patch[6];
    output_patch[1] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[1] - product_patch[2] - product_patch[3] + product_patch[5] - product_patch[6] - product_patch[7];
    output_patch[2] = product_patch[10] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[4] + product_patch[5] + product_patch[6] + product_patch[8] + product_patch[9];
    output_patch[3] = -product_patch[10] - product_patch[11] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[5] - product_patch[6] - product_patch[7] + product_patch[9];
    output_patch[4] = product_patch[12] + product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[24] + product_patch[25] + product_patch[26] + product_patch[28] + product_patch[29] + product_patch[30];
    output_patch[5] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[25] - product_patch[26] - product_patch[27] + product_patch[29] - product_patch[30] - product_patch[31];
    output_patch[6] = product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[28] + product_patch[29] + product_patch[30] + product_patch[32] + product_patch[33] + product_patch[34];
    output_patch[7] = product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[29] - product_patch[30] - product_patch[31] + product_patch[33] - product_patch[34] - product_patch[35];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (24, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x3x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[6] = {T(0)};
    for(int i = 0; i < 6; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[24] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[3] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[4] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[5] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[6] = ele[2];
    product_weight_patch[7] = ele[2];
    product_weight_patch[8] = ele[0] + ele[3];
    product_weight_patch[9] = ele[0] + ele[3];
    product_weight_patch[10] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[11] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[12] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[13] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[14] = ele[2] + ele[5];
    product_weight_patch[15] = ele[2] + ele[5];
    product_weight_patch[16] = ele[3];
    product_weight_patch[17] = ele[3];
    product_weight_patch[18] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[19] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[20] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[21] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[22] = ele[5];
    product_weight_patch[23] = ele[5];


    for(int i = 0; i < 24; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (24, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x3x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[24] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[8 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[8 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[24] = {T(0)};


    trans_input_patch[0] = input_patch[0] + input_patch[12] - input_patch[4] - input_patch[8];
    trans_input_patch[1] = input_patch[13] + input_patch[1] - input_patch[5] - input_patch[9];
    trans_input_patch[2] = -input_patch[10] - input_patch[12] + input_patch[2] + input_patch[4];
    trans_input_patch[3] = -input_patch[11] - input_patch[13] + input_patch[3] + input_patch[5];
    trans_input_patch[4] = input_patch[10] - input_patch[12] - input_patch[2] + input_patch[4];
    trans_input_patch[5] = input_patch[11] - input_patch[13] - input_patch[3] + input_patch[5];
    trans_input_patch[6] = -input_patch[10] + input_patch[14] + input_patch[2] - input_patch[6];
    trans_input_patch[7] = -input_patch[11] + input_patch[15] + input_patch[3] - input_patch[7];
    trans_input_patch[8] = -input_patch[12] + input_patch[8];
    trans_input_patch[9] = -input_patch[13] + input_patch[9];
    trans_input_patch[10] = input_patch[10] + input_patch[12];
    trans_input_patch[11] = input_patch[11] + input_patch[13];
    trans_input_patch[12] = -input_patch[10] + input_patch[12];
    trans_input_patch[13] = -input_patch[11] + input_patch[13];
    trans_input_patch[14] = input_patch[10] - input_patch[14];
    trans_input_patch[15] = input_patch[11] - input_patch[15];
    trans_input_patch[16] = input_patch[12] + input_patch[16] - input_patch[20] - input_patch[8];
    trans_input_patch[17] = input_patch[13] + input_patch[17] - input_patch[21] - input_patch[9];
    trans_input_patch[18] = -input_patch[10] - input_patch[12] + input_patch[18] + input_patch[20];
    trans_input_patch[19] = -input_patch[11] - input_patch[13] + input_patch[19] + input_patch[21];
    trans_input_patch[20] = input_patch[10] - input_patch[12] - input_patch[18] + input_patch[20];
    trans_input_patch[21] = input_patch[11] - input_patch[13] - input_patch[19] + input_patch[21];
    trans_input_patch[22] = -input_patch[10] + input_patch[14] + input_patch[18] - input_patch[22];
    trans_input_patch[23] = -input_patch[11] + input_patch[15] + input_patch[19] - input_patch[23];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 24; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (24, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x3x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[24] = {T(0)};

    for(int i = 0; i < 24; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[12] + product_patch[2] + product_patch[4] + product_patch[8];
    output_patch[1] = product_patch[11] + product_patch[13] + product_patch[1] + product_patch[3] + product_patch[5] + product_patch[9];
    output_patch[2] = product_patch[10] - product_patch[12] - product_patch[14] + product_patch[2] - product_patch[4] - product_patch[6];
    output_patch[3] = product_patch[11] - product_patch[13] - product_patch[15] + product_patch[3] - product_patch[5] - product_patch[7];
    output_patch[4] = product_patch[10] + product_patch[12] + product_patch[16] + product_patch[18] + product_patch[20] + product_patch[8];
    output_patch[5] = product_patch[11] + product_patch[13] + product_patch[17] + product_patch[19] + product_patch[21] + product_patch[9];
    output_patch[6] = product_patch[10] - product_patch[12] - product_patch[14] + product_patch[18] - product_patch[20] - product_patch[22];
    output_patch[7] = product_patch[11] - product_patch[13] - product_patch[15] + product_patch[19] - product_patch[21] - product_patch[23];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (12, C, K)
// wino_weight = (36, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x3x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[12] = {T(0)};
    for(int i = 0; i < 12; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[36] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[4] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[5] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[7] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[8] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[9] = ele[4];
    product_weight_patch[10] = ele[4] + ele[5];
    product_weight_patch[11] = ele[5];
    product_weight_patch[12] = ele[0] + ele[6];
    product_weight_patch[13] = ele[0] + ele[1] + ele[6] + ele[7];
    product_weight_patch[14] = ele[1] + ele[7];
    product_weight_patch[15] = T(1/2.)*(ele[0] + ele[10] + ele[2] + ele[4] + ele[6] + ele[8]);
    product_weight_patch[16] = T(1/2.)*(ele[0] + ele[10] + ele[11] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[17] = T(1/2.)*(ele[11] + ele[1] + ele[3] + ele[5] + ele[7] + ele[9]);
    product_weight_patch[18] = T(1/2.)*(ele[0] + ele[10] - ele[2] + ele[4] + ele[6] - ele[8]);
    product_weight_patch[19] = T(1/2.)*(ele[0] + ele[10] + ele[11] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5] + ele[6] + ele[7] - ele[8] - ele[9]);
    product_weight_patch[20] = T(1/2.)*(ele[11] + ele[1] - ele[3] + ele[5] + ele[7] - ele[9]);
    product_weight_patch[21] = ele[10] + ele[4];
    product_weight_patch[22] = ele[10] + ele[11] + ele[4] + ele[5];
    product_weight_patch[23] = ele[11] + ele[5];
    product_weight_patch[24] = ele[6];
    product_weight_patch[25] = ele[6] + ele[7];
    product_weight_patch[26] = ele[7];
    product_weight_patch[27] = T(1/2.)*(ele[10] + ele[6] + ele[8]);
    product_weight_patch[28] = T(1/2.)*(ele[10] + ele[11] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[29] = T(1/2.)*(ele[11] + ele[7] + ele[9]);
    product_weight_patch[30] = T(1/2.)*(ele[10] + ele[6] - ele[8]);
    product_weight_patch[31] = T(1/2.)*(ele[10] + ele[11] + ele[6] + ele[7] - ele[8] - ele[9]);
    product_weight_patch[32] = T(1/2.)*(ele[11] + ele[7] - ele[9]);
    product_weight_patch[33] = ele[10];
    product_weight_patch[34] = ele[10] + ele[11];
    product_weight_patch[35] = ele[11];


    for(int i = 0; i < 36; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (36, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x3x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[36] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[12 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[12 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[36] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[12] + input_patch[13] + input_patch[18] - input_patch[19] - input_patch[1] - input_patch[6] + input_patch[7];
    trans_input_patch[1] = -input_patch[13] + input_patch[19] + input_patch[1] - input_patch[7];
    trans_input_patch[2] = input_patch[13] - input_patch[14] - input_patch[19] - input_patch[1] + input_patch[20] + input_patch[2] + input_patch[7] - input_patch[8];
    trans_input_patch[3] = -input_patch[15] + input_patch[16] - input_patch[18] + input_patch[19] + input_patch[3] - input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[4] = -input_patch[16] - input_patch[19] + input_patch[4] + input_patch[7];
    trans_input_patch[5] = input_patch[16] - input_patch[17] + input_patch[19] - input_patch[20] - input_patch[4] + input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[6] = input_patch[15] - input_patch[16] - input_patch[18] + input_patch[19] - input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[7] = input_patch[16] - input_patch[19] - input_patch[4] + input_patch[7];
    trans_input_patch[8] = -input_patch[16] + input_patch[17] + input_patch[19] - input_patch[20] + input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[9] = input_patch[10] - input_patch[15] + input_patch[16] + input_patch[21] - input_patch[22] + input_patch[3] - input_patch[4] - input_patch[9];
    trans_input_patch[10] = -input_patch[10] - input_patch[16] + input_patch[22] + input_patch[4];
    trans_input_patch[11] = input_patch[10] - input_patch[11] + input_patch[16] - input_patch[17] - input_patch[22] + input_patch[23] - input_patch[4] + input_patch[5];
    trans_input_patch[12] = input_patch[12] - input_patch[13] - input_patch[18] + input_patch[19];
    trans_input_patch[13] = input_patch[13] - input_patch[19];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[19] - input_patch[20];
    trans_input_patch[15] = input_patch[15] - input_patch[16] + input_patch[18] - input_patch[19];
    trans_input_patch[16] = input_patch[16] + input_patch[19];
    trans_input_patch[17] = -input_patch[16] + input_patch[17] - input_patch[19] + input_patch[20];
    trans_input_patch[18] = -input_patch[15] + input_patch[16] + input_patch[18] - input_patch[19];
    trans_input_patch[19] = -input_patch[16] + input_patch[19];
    trans_input_patch[20] = input_patch[16] - input_patch[17] - input_patch[19] + input_patch[20];
    trans_input_patch[21] = input_patch[15] - input_patch[16] - input_patch[21] + input_patch[22];
    trans_input_patch[22] = input_patch[16] - input_patch[22];
    trans_input_patch[23] = -input_patch[16] + input_patch[17] + input_patch[22] - input_patch[23];
    trans_input_patch[24] = -input_patch[12] + input_patch[13] + input_patch[18] - input_patch[19] + input_patch[24] - input_patch[25] - input_patch[30] + input_patch[31];
    trans_input_patch[25] = -input_patch[13] + input_patch[19] + input_patch[25] - input_patch[31];
    trans_input_patch[26] = input_patch[13] - input_patch[14] - input_patch[19] + input_patch[20] - input_patch[25] + input_patch[26] + input_patch[31] - input_patch[32];
    trans_input_patch[27] = -input_patch[15] + input_patch[16] - input_patch[18] + input_patch[19] + input_patch[27] - input_patch[28] + input_patch[30] - input_patch[31];
    trans_input_patch[28] = -input_patch[16] - input_patch[19] + input_patch[28] + input_patch[31];
    trans_input_patch[29] = input_patch[16] - input_patch[17] + input_patch[19] - input_patch[20] - input_patch[28] + input_patch[29] - input_patch[31] + input_patch[32];
    trans_input_patch[30] = input_patch[15] - input_patch[16] - input_patch[18] + input_patch[19] - input_patch[27] + input_patch[28] + input_patch[30] - input_patch[31];
    trans_input_patch[31] = input_patch[16] - input_patch[19] - input_patch[28] + input_patch[31];
    trans_input_patch[32] = -input_patch[16] + input_patch[17] + input_patch[19] - input_patch[20] + input_patch[28] - input_patch[29] - input_patch[31] + input_patch[32];
    trans_input_patch[33] = -input_patch[15] + input_patch[16] + input_patch[21] - input_patch[22] + input_patch[27] - input_patch[28] - input_patch[33] + input_patch[34];
    trans_input_patch[34] = -input_patch[16] + input_patch[22] + input_patch[28] - input_patch[34];
    trans_input_patch[35] = input_patch[16] - input_patch[17] - input_patch[22] + input_patch[23] - input_patch[28] + input_patch[29] + input_patch[34] - input_patch[35];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 36; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (36, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x3x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[36] = {T(0)};

    for(int i = 0; i < 36; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[18] + product_patch[19] + product_patch[1] + product_patch[3] + product_patch[4] + product_patch[6] + product_patch[7];
    output_patch[1] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[19] + product_patch[1] + product_patch[20] + product_patch[2] + product_patch[4] + product_patch[5] + product_patch[7] + product_patch[8];
    output_patch[2] = -product_patch[10] + product_patch[15] + product_patch[16] - product_patch[18] - product_patch[19] - product_patch[21] - product_patch[22] + product_patch[3] + product_patch[4] - product_patch[6] - product_patch[7] - product_patch[9];
    output_patch[3] = -product_patch[10] - product_patch[11] + product_patch[16] + product_patch[17] - product_patch[19] - product_patch[20] - product_patch[22] - product_patch[23] + product_patch[4] + product_patch[5] - product_patch[7] - product_patch[8];
    output_patch[4] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[18] + product_patch[19] + product_patch[24] + product_patch[25] + product_patch[27] + product_patch[28] + product_patch[30] + product_patch[31];
    output_patch[5] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[19] + product_patch[20] + product_patch[25] + product_patch[26] + product_patch[28] + product_patch[29] + product_patch[31] + product_patch[32];
    output_patch[6] = product_patch[15] + product_patch[16] - product_patch[18] - product_patch[19] - product_patch[21] - product_patch[22] + product_patch[27] + product_patch[28] - product_patch[30] - product_patch[31] - product_patch[33] - product_patch[34];
    output_patch[7] = product_patch[16] + product_patch[17] - product_patch[19] - product_patch[20] - product_patch[22] - product_patch[23] + product_patch[28] + product_patch[29] - product_patch[31] - product_patch[32] - product_patch[34] - product_patch[35];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (18, C, K)
// wino_weight = (48, C, K)
template <typename T>
__global__ void wNorm2WinoTransform2x3x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[18] = {T(0)};
    for(int i = 0; i < 18; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[48] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[5] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[6] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[7] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[8] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[9] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[10] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[11] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[12] = ele[6];
    product_weight_patch[13] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[14] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[15] = ele[8];
    product_weight_patch[16] = ele[0] + ele[9];
    product_weight_patch[17] = T(1/2.)*(ele[0] + ele[10] + ele[11] + ele[1] + ele[2] + ele[9]);
    product_weight_patch[18] = T(1/2.)*(ele[0] - ele[10] + ele[11] - ele[1] + ele[2] + ele[9]);
    product_weight_patch[19] = ele[11] + ele[2];
    product_weight_patch[20] = T(1/2.)*(ele[0] + ele[12] + ele[15] + ele[3] + ele[6] + ele[9]);
    product_weight_patch[21] = T(1/4.)*(ele[0] + ele[10] + ele[11] + ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[22] = T(1/4.)*(ele[0] - ele[10] + ele[11] + ele[12] - ele[13] + ele[14] + ele[15] - ele[16] + ele[17] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[23] = T(1/2.)*(ele[11] + ele[14] + ele[17] + ele[2] + ele[5] + ele[8]);
    product_weight_patch[24] = T(1/2.)*(ele[0] - ele[12] + ele[15] - ele[3] + ele[6] + ele[9]);
    product_weight_patch[25] = T(1/4.)*(ele[0] + ele[10] + ele[11] - ele[12] - ele[13] - ele[14] + ele[15] + ele[16] + ele[17] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[26] = T(1/4.)*(ele[0] - ele[10] + ele[11] - ele[12] + ele[13] - ele[14] + ele[15] - ele[16] + ele[17] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[27] = T(1/2.)*(ele[11] - ele[14] + ele[17] + ele[2] - ele[5] + ele[8]);
    product_weight_patch[28] = ele[15] + ele[6];
    product_weight_patch[29] = T(1/2.)*(ele[15] + ele[16] + ele[17] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[30] = T(1/2.)*(ele[15] - ele[16] + ele[17] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[31] = ele[17] + ele[8];
    product_weight_patch[32] = ele[9];
    product_weight_patch[33] = T(1/2.)*(ele[10] + ele[11] + ele[9]);
    product_weight_patch[34] = T(1/2.)*(-ele[10] + ele[11] + ele[9]);
    product_weight_patch[35] = ele[11];
    product_weight_patch[36] = T(1/2.)*(ele[12] + ele[15] + ele[9]);
    product_weight_patch[37] = T(1/4.)*(ele[10] + ele[11] + ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17] + ele[9]);
    product_weight_patch[38] = T(1/4.)*(-ele[10] + ele[11] + ele[12] - ele[13] + ele[14] + ele[15] - ele[16] + ele[17] + ele[9]);
    product_weight_patch[39] = T(1/2.)*(ele[11] + ele[14] + ele[17]);
    product_weight_patch[40] = T(1/2.)*(-ele[12] + ele[15] + ele[9]);
    product_weight_patch[41] = T(1/4.)*(ele[10] + ele[11] - ele[12] - ele[13] - ele[14] + ele[15] + ele[16] + ele[17] + ele[9]);
    product_weight_patch[42] = T(1/4.)*(-ele[10] + ele[11] - ele[12] + ele[13] - ele[14] + ele[15] - ele[16] + ele[17] + ele[9]);
    product_weight_patch[43] = T(1/2.)*(ele[11] - ele[14] + ele[17]);
    product_weight_patch[44] = ele[15];
    product_weight_patch[45] = T(1/2.)*(ele[15] + ele[16] + ele[17]);
    product_weight_patch[46] = T(1/2.)*(ele[15] - ele[16] + ele[17]);
    product_weight_patch[47] = ele[17];


    for(int i = 0; i < 48; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (48, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform2x3x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 3) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[48] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[16 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[16 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[48] = {T(0)};


    trans_input_patch[0] = input_patch[0] + input_patch[10] - input_patch[16] + input_patch[18] + input_patch[24] - input_patch[26] - input_patch[2] - input_patch[8];
    trans_input_patch[1] = -input_patch[10] - input_patch[17] - input_patch[18] + input_patch[1] + input_patch[25] + input_patch[26] + input_patch[2] - input_patch[9];
    trans_input_patch[2] = -input_patch[10] + input_patch[17] - input_patch[18] - input_patch[1] - input_patch[25] + input_patch[26] + input_patch[2] + input_patch[9];
    trans_input_patch[3] = input_patch[11] - input_patch[17] + input_patch[19] + input_patch[1] + input_patch[25] - input_patch[27] - input_patch[3] - input_patch[9];
    trans_input_patch[4] = -input_patch[10] - input_patch[20] + input_patch[22] - input_patch[24] + input_patch[26] + input_patch[4] - input_patch[6] + input_patch[8];
    trans_input_patch[5] = input_patch[10] - input_patch[21] - input_patch[22] - input_patch[25] - input_patch[26] + input_patch[5] + input_patch[6] + input_patch[9];
    trans_input_patch[6] = input_patch[10] + input_patch[21] - input_patch[22] + input_patch[25] - input_patch[26] - input_patch[5] + input_patch[6] - input_patch[9];
    trans_input_patch[7] = -input_patch[11] - input_patch[21] + input_patch[23] - input_patch[25] + input_patch[27] + input_patch[5] - input_patch[7] + input_patch[9];
    trans_input_patch[8] = -input_patch[10] + input_patch[20] - input_patch[22] - input_patch[24] + input_patch[26] - input_patch[4] + input_patch[6] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[21] + input_patch[22] - input_patch[25] - input_patch[26] - input_patch[5] - input_patch[6] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[21] + input_patch[22] + input_patch[25] - input_patch[26] + input_patch[5] - input_patch[6] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] + input_patch[21] - input_patch[23] - input_patch[25] + input_patch[27] - input_patch[5] + input_patch[7] + input_patch[9];
    trans_input_patch[12] = -input_patch[12] + input_patch[14] - input_patch[20] + input_patch[22] + input_patch[28] - input_patch[30] + input_patch[4] - input_patch[6];
    trans_input_patch[13] = -input_patch[13] - input_patch[14] - input_patch[21] - input_patch[22] + input_patch[29] + input_patch[30] + input_patch[5] + input_patch[6];
    trans_input_patch[14] = input_patch[13] - input_patch[14] + input_patch[21] - input_patch[22] - input_patch[29] + input_patch[30] - input_patch[5] + input_patch[6];
    trans_input_patch[15] = -input_patch[13] + input_patch[15] - input_patch[21] + input_patch[23] + input_patch[29] - input_patch[31] + input_patch[5] - input_patch[7];
    trans_input_patch[16] = input_patch[16] - input_patch[18] - input_patch[24] + input_patch[26];
    trans_input_patch[17] = input_patch[17] + input_patch[18] - input_patch[25] - input_patch[26];
    trans_input_patch[18] = -input_patch[17] + input_patch[18] + input_patch[25] - input_patch[26];
    trans_input_patch[19] = input_patch[17] - input_patch[19] - input_patch[25] + input_patch[27];
    trans_input_patch[20] = input_patch[20] - input_patch[22] + input_patch[24] - input_patch[26];
    trans_input_patch[21] = input_patch[21] + input_patch[22] + input_patch[25] + input_patch[26];
    trans_input_patch[22] = -input_patch[21] + input_patch[22] - input_patch[25] + input_patch[26];
    trans_input_patch[23] = input_patch[21] - input_patch[23] + input_patch[25] - input_patch[27];
    trans_input_patch[24] = -input_patch[20] + input_patch[22] + input_patch[24] - input_patch[26];
    trans_input_patch[25] = -input_patch[21] - input_patch[22] + input_patch[25] + input_patch[26];
    trans_input_patch[26] = input_patch[21] - input_patch[22] - input_patch[25] + input_patch[26];
    trans_input_patch[27] = -input_patch[21] + input_patch[23] + input_patch[25] - input_patch[27];
    trans_input_patch[28] = input_patch[20] - input_patch[22] - input_patch[28] + input_patch[30];
    trans_input_patch[29] = input_patch[21] + input_patch[22] - input_patch[29] - input_patch[30];
    trans_input_patch[30] = -input_patch[21] + input_patch[22] + input_patch[29] - input_patch[30];
    trans_input_patch[31] = input_patch[21] - input_patch[23] - input_patch[29] + input_patch[31];
    trans_input_patch[32] = -input_patch[16] + input_patch[18] + input_patch[24] - input_patch[26] + input_patch[32] - input_patch[34] - input_patch[40] + input_patch[42];
    trans_input_patch[33] = -input_patch[17] - input_patch[18] + input_patch[25] + input_patch[26] + input_patch[33] + input_patch[34] - input_patch[41] - input_patch[42];
    trans_input_patch[34] = input_patch[17] - input_patch[18] - input_patch[25] + input_patch[26] - input_patch[33] + input_patch[34] + input_patch[41] - input_patch[42];
    trans_input_patch[35] = -input_patch[17] + input_patch[19] + input_patch[25] - input_patch[27] + input_patch[33] - input_patch[35] - input_patch[41] + input_patch[43];
    trans_input_patch[36] = -input_patch[20] + input_patch[22] - input_patch[24] + input_patch[26] + input_patch[36] - input_patch[38] + input_patch[40] - input_patch[42];
    trans_input_patch[37] = -input_patch[21] - input_patch[22] - input_patch[25] - input_patch[26] + input_patch[37] + input_patch[38] + input_patch[41] + input_patch[42];
    trans_input_patch[38] = input_patch[21] - input_patch[22] + input_patch[25] - input_patch[26] - input_patch[37] + input_patch[38] - input_patch[41] + input_patch[42];
    trans_input_patch[39] = -input_patch[21] + input_patch[23] - input_patch[25] + input_patch[27] + input_patch[37] - input_patch[39] + input_patch[41] - input_patch[43];
    trans_input_patch[40] = input_patch[20] - input_patch[22] - input_patch[24] + input_patch[26] - input_patch[36] + input_patch[38] + input_patch[40] - input_patch[42];
    trans_input_patch[41] = input_patch[21] + input_patch[22] - input_patch[25] - input_patch[26] - input_patch[37] - input_patch[38] + input_patch[41] + input_patch[42];
    trans_input_patch[42] = -input_patch[21] + input_patch[22] + input_patch[25] - input_patch[26] + input_patch[37] - input_patch[38] - input_patch[41] + input_patch[42];
    trans_input_patch[43] = input_patch[21] - input_patch[23] - input_patch[25] + input_patch[27] - input_patch[37] + input_patch[39] + input_patch[41] - input_patch[43];
    trans_input_patch[44] = -input_patch[20] + input_patch[22] + input_patch[28] - input_patch[30] + input_patch[36] - input_patch[38] - input_patch[44] + input_patch[46];
    trans_input_patch[45] = -input_patch[21] - input_patch[22] + input_patch[29] + input_patch[30] + input_patch[37] + input_patch[38] - input_patch[45] - input_patch[46];
    trans_input_patch[46] = input_patch[21] - input_patch[22] - input_patch[29] + input_patch[30] - input_patch[37] + input_patch[38] + input_patch[45] - input_patch[46];
    trans_input_patch[47] = -input_patch[21] + input_patch[23] + input_patch[29] - input_patch[31] + input_patch[37] - input_patch[39] - input_patch[45] + input_patch[47];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 48; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (48, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform2x3x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[48] = {T(0)};

    for(int i = 0; i < 48; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[1] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[24] + product_patch[25] + product_patch[26] + product_patch[2] + product_patch[4] + product_patch[5] + product_patch[6] + product_patch[8] + product_patch[9];
    output_patch[1] = -product_patch[10] - product_patch[11] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[1] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[25] - product_patch[26] - product_patch[27] - product_patch[2] - product_patch[3] + product_patch[5] - product_patch[6] - product_patch[7] + product_patch[9];
    output_patch[2] = -product_patch[10] - product_patch[12] - product_patch[13] - product_patch[14] + product_patch[20] + product_patch[21] + product_patch[22] - product_patch[24] - product_patch[25] - product_patch[26] - product_patch[28] - product_patch[29] - product_patch[30] + product_patch[4] + product_patch[5] + product_patch[6] - product_patch[8] - product_patch[9];
    output_patch[3] = product_patch[10] + product_patch[11] - product_patch[13] + product_patch[14] + product_patch[15] + product_patch[21] - product_patch[22] - product_patch[23] - product_patch[25] + product_patch[26] + product_patch[27] - product_patch[29] + product_patch[30] + product_patch[31] + product_patch[5] - product_patch[6] - product_patch[7] - product_patch[9];
    output_patch[4] = product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[24] + product_patch[25] + product_patch[26] + product_patch[32] + product_patch[33] + product_patch[34] + product_patch[36] + product_patch[37] + product_patch[38] + product_patch[40] + product_patch[41] + product_patch[42];
    output_patch[5] = product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[25] - product_patch[26] - product_patch[27] + product_patch[33] - product_patch[34] - product_patch[35] + product_patch[37] - product_patch[38] - product_patch[39] + product_patch[41] - product_patch[42] - product_patch[43];
    output_patch[6] = product_patch[20] + product_patch[21] + product_patch[22] - product_patch[24] - product_patch[25] - product_patch[26] - product_patch[28] - product_patch[29] - product_patch[30] + product_patch[36] + product_patch[37] + product_patch[38] - product_patch[40] - product_patch[41] - product_patch[42] - product_patch[44] - product_patch[45] - product_patch[46];
    output_patch[7] = product_patch[21] - product_patch[22] - product_patch[23] - product_patch[25] + product_patch[26] + product_patch[27] - product_patch[29] + product_patch[30] + product_patch[31] + product_patch[37] - product_patch[38] - product_patch[39] - product_patch[41] + product_patch[42] + product_patch[43] - product_patch[45] + product_patch[46] + product_patch[47];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (3, C, K)
// wino_weight = (16, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x1x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[3] = {T(0)};
    for(int i = 0; i < 3; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[16] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = ele[0];
    product_weight_patch[3] = ele[0];
    product_weight_patch[4] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[5] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[6] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[7] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[8] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[9] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[10] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[11] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[12] = ele[2];
    product_weight_patch[13] = ele[2];
    product_weight_patch[14] = ele[2];
    product_weight_patch[15] = ele[2];


    for(int i = 0; i < 16; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (16, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x1x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[16] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[4 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[4 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[16] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[8];
    trans_input_patch[1] = input_patch[1] - input_patch[9];
    trans_input_patch[2] = -input_patch[10] + input_patch[2];
    trans_input_patch[3] = -input_patch[11] + input_patch[3];
    trans_input_patch[4] = input_patch[4] + input_patch[8];
    trans_input_patch[5] = input_patch[5] + input_patch[9];
    trans_input_patch[6] = input_patch[10] + input_patch[6];
    trans_input_patch[7] = input_patch[11] + input_patch[7];
    trans_input_patch[8] = -input_patch[4] + input_patch[8];
    trans_input_patch[9] = -input_patch[5] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[6];
    trans_input_patch[11] = input_patch[11] - input_patch[7];
    trans_input_patch[12] = -input_patch[12] + input_patch[4];
    trans_input_patch[13] = -input_patch[13] + input_patch[5];
    trans_input_patch[14] = -input_patch[14] + input_patch[6];
    trans_input_patch[15] = -input_patch[15] + input_patch[7];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 16; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (16, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x1x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[16] = {T(0)};

    for(int i = 0; i < 16; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[4] + product_patch[8];
    output_patch[1] = product_patch[1] + product_patch[5] + product_patch[9];
    output_patch[2] = product_patch[10] + product_patch[2] + product_patch[6];
    output_patch[3] = product_patch[11] + product_patch[3] + product_patch[7];
    output_patch[4] = -product_patch[12] + product_patch[4] - product_patch[8];
    output_patch[5] = -product_patch[13] + product_patch[5] - product_patch[9];
    output_patch[6] = -product_patch[10] - product_patch[14] + product_patch[6];
    output_patch[7] = -product_patch[11] - product_patch[15] + product_patch[7];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (24, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x1x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[6] = {T(0)};
    for(int i = 0; i < 6; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[24] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = ele[0];
    product_weight_patch[4] = ele[0] + ele[1];
    product_weight_patch[5] = ele[1];
    product_weight_patch[6] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[7] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[8] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[9] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[10] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[11] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[12] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[13] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[14] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[15] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[16] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[17] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[18] = ele[4];
    product_weight_patch[19] = ele[4] + ele[5];
    product_weight_patch[20] = ele[5];
    product_weight_patch[21] = ele[4];
    product_weight_patch[22] = ele[4] + ele[5];
    product_weight_patch[23] = ele[5];


    for(int i = 0; i < 24; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (24, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x1x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[24] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[6 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[6 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[24] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[12] + input_patch[13] - input_patch[1];
    trans_input_patch[1] = -input_patch[13] + input_patch[1];
    trans_input_patch[2] = input_patch[13] - input_patch[14] - input_patch[1] + input_patch[2];
    trans_input_patch[3] = -input_patch[15] + input_patch[16] + input_patch[3] - input_patch[4];
    trans_input_patch[4] = -input_patch[16] + input_patch[4];
    trans_input_patch[5] = input_patch[16] - input_patch[17] - input_patch[4] + input_patch[5];
    trans_input_patch[6] = input_patch[12] - input_patch[13] + input_patch[6] - input_patch[7];
    trans_input_patch[7] = input_patch[13] + input_patch[7];
    trans_input_patch[8] = -input_patch[13] + input_patch[14] - input_patch[7] + input_patch[8];
    trans_input_patch[9] = -input_patch[10] + input_patch[15] - input_patch[16] + input_patch[9];
    trans_input_patch[10] = input_patch[10] + input_patch[16];
    trans_input_patch[11] = -input_patch[10] + input_patch[11] - input_patch[16] + input_patch[17];
    trans_input_patch[12] = input_patch[12] - input_patch[13] - input_patch[6] + input_patch[7];
    trans_input_patch[13] = input_patch[13] - input_patch[7];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[7] - input_patch[8];
    trans_input_patch[15] = input_patch[10] + input_patch[15] - input_patch[16] - input_patch[9];
    trans_input_patch[16] = -input_patch[10] + input_patch[16];
    trans_input_patch[17] = input_patch[10] - input_patch[11] - input_patch[16] + input_patch[17];
    trans_input_patch[18] = -input_patch[18] + input_patch[19] + input_patch[6] - input_patch[7];
    trans_input_patch[19] = -input_patch[19] + input_patch[7];
    trans_input_patch[20] = input_patch[19] - input_patch[20] - input_patch[7] + input_patch[8];
    trans_input_patch[21] = -input_patch[10] - input_patch[21] + input_patch[22] + input_patch[9];
    trans_input_patch[22] = input_patch[10] - input_patch[22];
    trans_input_patch[23] = -input_patch[10] + input_patch[11] + input_patch[22] - input_patch[23];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 24; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (24, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x1x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[24] = {T(0)};

    for(int i = 0; i < 24; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[12] + product_patch[13] + product_patch[1] + product_patch[6] + product_patch[7];
    output_patch[1] = product_patch[13] + product_patch[14] + product_patch[1] + product_patch[2] + product_patch[7] + product_patch[8];
    output_patch[2] = product_patch[10] + product_patch[15] + product_patch[16] + product_patch[3] + product_patch[4] + product_patch[9];
    output_patch[3] = product_patch[10] + product_patch[11] + product_patch[16] + product_patch[17] + product_patch[4] + product_patch[5];
    output_patch[4] = -product_patch[12] - product_patch[13] - product_patch[18] - product_patch[19] + product_patch[6] + product_patch[7];
    output_patch[5] = -product_patch[13] - product_patch[14] - product_patch[19] - product_patch[20] + product_patch[7] + product_patch[8];
    output_patch[6] = product_patch[10] - product_patch[15] - product_patch[16] - product_patch[21] - product_patch[22] + product_patch[9];
    output_patch[7] = product_patch[10] + product_patch[11] - product_patch[16] - product_patch[17] - product_patch[22] - product_patch[23];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (9, C, K)
// wino_weight = (32, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x1x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[9] = {T(0)};
    for(int i = 0; i < 9; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[32] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = ele[0];
    product_weight_patch[5] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[7] = ele[2];
    product_weight_patch[8] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[9] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[10] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[11] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[12] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[13] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[14] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[15] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[16] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[17] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[18] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[19] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[20] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[21] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[22] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[23] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[24] = ele[6];
    product_weight_patch[25] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[26] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[27] = ele[8];
    product_weight_patch[28] = ele[6];
    product_weight_patch[29] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[30] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[31] = ele[8];


    for(int i = 0; i < 32; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (32, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x1x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 2) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[32] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[8 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[8 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[32] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[16] + input_patch[18] - input_patch[2];
    trans_input_patch[1] = -input_patch[17] - input_patch[18] + input_patch[1] + input_patch[2];
    trans_input_patch[2] = input_patch[17] - input_patch[18] - input_patch[1] + input_patch[2];
    trans_input_patch[3] = -input_patch[17] + input_patch[19] + input_patch[1] - input_patch[3];
    trans_input_patch[4] = -input_patch[20] + input_patch[22] + input_patch[4] - input_patch[6];
    trans_input_patch[5] = -input_patch[21] - input_patch[22] + input_patch[5] + input_patch[6];
    trans_input_patch[6] = input_patch[21] - input_patch[22] - input_patch[5] + input_patch[6];
    trans_input_patch[7] = -input_patch[21] + input_patch[23] + input_patch[5] - input_patch[7];
    trans_input_patch[8] = -input_patch[10] + input_patch[16] - input_patch[18] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[17] + input_patch[18] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[17] + input_patch[18] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] + input_patch[17] - input_patch[19] + input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14] + input_patch[20] - input_patch[22];
    trans_input_patch[13] = input_patch[13] + input_patch[14] + input_patch[21] + input_patch[22];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] - input_patch[21] + input_patch[22];
    trans_input_patch[15] = input_patch[13] - input_patch[15] + input_patch[21] - input_patch[23];
    trans_input_patch[16] = input_patch[10] + input_patch[16] - input_patch[18] - input_patch[8];
    trans_input_patch[17] = -input_patch[10] + input_patch[17] + input_patch[18] - input_patch[9];
    trans_input_patch[18] = -input_patch[10] - input_patch[17] + input_patch[18] + input_patch[9];
    trans_input_patch[19] = input_patch[11] + input_patch[17] - input_patch[19] - input_patch[9];
    trans_input_patch[20] = -input_patch[12] + input_patch[14] + input_patch[20] - input_patch[22];
    trans_input_patch[21] = -input_patch[13] - input_patch[14] + input_patch[21] + input_patch[22];
    trans_input_patch[22] = input_patch[13] - input_patch[14] - input_patch[21] + input_patch[22];
    trans_input_patch[23] = -input_patch[13] + input_patch[15] + input_patch[21] - input_patch[23];
    trans_input_patch[24] = -input_patch[10] - input_patch[24] + input_patch[26] + input_patch[8];
    trans_input_patch[25] = input_patch[10] - input_patch[25] - input_patch[26] + input_patch[9];
    trans_input_patch[26] = input_patch[10] + input_patch[25] - input_patch[26] - input_patch[9];
    trans_input_patch[27] = -input_patch[11] - input_patch[25] + input_patch[27] + input_patch[9];
    trans_input_patch[28] = input_patch[12] - input_patch[14] - input_patch[28] + input_patch[30];
    trans_input_patch[29] = input_patch[13] + input_patch[14] - input_patch[29] - input_patch[30];
    trans_input_patch[30] = -input_patch[13] + input_patch[14] + input_patch[29] - input_patch[30];
    trans_input_patch[31] = input_patch[13] - input_patch[15] - input_patch[29] + input_patch[31];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 32; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (32, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x1x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[32] = {T(0)};

    for(int i = 0; i < 32; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[1] + product_patch[2] + product_patch[8] + product_patch[9];
    output_patch[1] = -product_patch[10] - product_patch[11] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[1] - product_patch[2] - product_patch[3] + product_patch[9];
    output_patch[2] = product_patch[12] + product_patch[13] + product_patch[14] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[4] + product_patch[5] + product_patch[6];
    output_patch[3] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[5] - product_patch[6] - product_patch[7];
    output_patch[4] = product_patch[10] - product_patch[16] - product_patch[17] - product_patch[18] - product_patch[24] - product_patch[25] - product_patch[26] + product_patch[8] + product_patch[9];
    output_patch[5] = -product_patch[10] - product_patch[11] - product_patch[17] + product_patch[18] + product_patch[19] - product_patch[25] + product_patch[26] + product_patch[27] + product_patch[9];
    output_patch[6] = product_patch[12] + product_patch[13] + product_patch[14] - product_patch[20] - product_patch[21] - product_patch[22] - product_patch[28] - product_patch[29] - product_patch[30];
    output_patch[7] = product_patch[13] - product_patch[14] - product_patch[15] - product_patch[21] + product_patch[22] + product_patch[23] - product_patch[29] + product_patch[30] + product_patch[31];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (6, C, K)
// wino_weight = (24, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x2x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[6] = {T(0)};
    for(int i = 0; i < 6; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[24] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = ele[0] + ele[1];
    product_weight_patch[3] = ele[0] + ele[1];
    product_weight_patch[4] = ele[1];
    product_weight_patch[5] = ele[1];
    product_weight_patch[6] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[7] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[8] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[9] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[10] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[11] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[12] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[13] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[14] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[15] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[16] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[17] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[18] = ele[4];
    product_weight_patch[19] = ele[4];
    product_weight_patch[20] = ele[4] + ele[5];
    product_weight_patch[21] = ele[4] + ele[5];
    product_weight_patch[22] = ele[5];
    product_weight_patch[23] = ele[5];


    for(int i = 0; i < 24; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (24, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x2x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[24] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[6 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[6 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[24] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[12] + input_patch[14] - input_patch[2];
    trans_input_patch[1] = -input_patch[13] + input_patch[15] + input_patch[1] - input_patch[3];
    trans_input_patch[2] = -input_patch[14] + input_patch[2];
    trans_input_patch[3] = -input_patch[15] + input_patch[3];
    trans_input_patch[4] = input_patch[14] - input_patch[16] - input_patch[2] + input_patch[4];
    trans_input_patch[5] = input_patch[15] - input_patch[17] - input_patch[3] + input_patch[5];
    trans_input_patch[6] = input_patch[12] - input_patch[14] + input_patch[6] - input_patch[8];
    trans_input_patch[7] = input_patch[13] - input_patch[15] + input_patch[7] - input_patch[9];
    trans_input_patch[8] = input_patch[14] + input_patch[8];
    trans_input_patch[9] = input_patch[15] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[14] + input_patch[16] - input_patch[8];
    trans_input_patch[11] = input_patch[11] - input_patch[15] + input_patch[17] - input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14] - input_patch[6] + input_patch[8];
    trans_input_patch[13] = input_patch[13] - input_patch[15] - input_patch[7] + input_patch[9];
    trans_input_patch[14] = input_patch[14] - input_patch[8];
    trans_input_patch[15] = input_patch[15] - input_patch[9];
    trans_input_patch[16] = -input_patch[10] - input_patch[14] + input_patch[16] + input_patch[8];
    trans_input_patch[17] = -input_patch[11] - input_patch[15] + input_patch[17] + input_patch[9];
    trans_input_patch[18] = -input_patch[18] + input_patch[20] + input_patch[6] - input_patch[8];
    trans_input_patch[19] = -input_patch[19] + input_patch[21] + input_patch[7] - input_patch[9];
    trans_input_patch[20] = -input_patch[20] + input_patch[8];
    trans_input_patch[21] = -input_patch[21] + input_patch[9];
    trans_input_patch[22] = input_patch[10] + input_patch[20] - input_patch[22] - input_patch[8];
    trans_input_patch[23] = input_patch[11] + input_patch[21] - input_patch[23] - input_patch[9];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 24; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (24, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x2x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[24] = {T(0)};

    for(int i = 0; i < 24; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[12] + product_patch[14] + product_patch[2] + product_patch[6] + product_patch[8];
    output_patch[1] = product_patch[13] + product_patch[15] + product_patch[1] + product_patch[3] + product_patch[7] + product_patch[9];
    output_patch[2] = product_patch[10] + product_patch[14] + product_patch[16] + product_patch[2] + product_patch[4] + product_patch[8];
    output_patch[3] = product_patch[11] + product_patch[15] + product_patch[17] + product_patch[3] + product_patch[5] + product_patch[9];
    output_patch[4] = -product_patch[12] - product_patch[14] - product_patch[18] - product_patch[20] + product_patch[6] + product_patch[8];
    output_patch[5] = -product_patch[13] - product_patch[15] - product_patch[19] - product_patch[21] + product_patch[7] + product_patch[9];
    output_patch[6] = product_patch[10] - product_patch[14] - product_patch[16] - product_patch[20] - product_patch[22] + product_patch[8];
    output_patch[7] = product_patch[11] - product_patch[15] - product_patch[17] - product_patch[21] - product_patch[23] + product_patch[9];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (12, C, K)
// wino_weight = (36, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x2x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[12] = {T(0)};
    for(int i = 0; i < 12; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[36] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = ele[0] + ele[2];
    product_weight_patch[4] = ele[0] + ele[1] + ele[2] + ele[3];
    product_weight_patch[5] = ele[1] + ele[3];
    product_weight_patch[6] = ele[2];
    product_weight_patch[7] = ele[2] + ele[3];
    product_weight_patch[8] = ele[3];
    product_weight_patch[9] = T(1/2.)*(ele[0] + ele[4] + ele[8]);
    product_weight_patch[10] = T(1/2.)*(ele[0] + ele[1] + ele[4] + ele[5] + ele[8] + ele[9]);
    product_weight_patch[11] = T(1/2.)*(ele[1] + ele[5] + ele[9]);
    product_weight_patch[12] = T(1/2.)*(ele[0] + ele[10] + ele[2] + ele[4] + ele[6] + ele[8]);
    product_weight_patch[13] = T(1/2.)*(ele[0] + ele[10] + ele[11] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[14] = T(1/2.)*(ele[11] + ele[1] + ele[3] + ele[5] + ele[7] + ele[9]);
    product_weight_patch[15] = T(1/2.)*(ele[10] + ele[2] + ele[6]);
    product_weight_patch[16] = T(1/2.)*(ele[10] + ele[11] + ele[2] + ele[3] + ele[6] + ele[7]);
    product_weight_patch[17] = T(1/2.)*(ele[11] + ele[3] + ele[7]);
    product_weight_patch[18] = T(1/2.)*(ele[0] - ele[4] + ele[8]);
    product_weight_patch[19] = T(1/2.)*(ele[0] + ele[1] - ele[4] - ele[5] + ele[8] + ele[9]);
    product_weight_patch[20] = T(1/2.)*(ele[1] - ele[5] + ele[9]);
    product_weight_patch[21] = T(1/2.)*(ele[0] + ele[10] + ele[2] - ele[4] - ele[6] + ele[8]);
    product_weight_patch[22] = T(1/2.)*(ele[0] + ele[10] + ele[11] + ele[1] + ele[2] + ele[3] - ele[4] - ele[5] - ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[23] = T(1/2.)*(ele[11] + ele[1] + ele[3] - ele[5] - ele[7] + ele[9]);
    product_weight_patch[24] = T(1/2.)*(ele[10] + ele[2] - ele[6]);
    product_weight_patch[25] = T(1/2.)*(ele[10] + ele[11] + ele[2] + ele[3] - ele[6] - ele[7]);
    product_weight_patch[26] = T(1/2.)*(ele[11] + ele[3] - ele[7]);
    product_weight_patch[27] = ele[8];
    product_weight_patch[28] = ele[8] + ele[9];
    product_weight_patch[29] = ele[9];
    product_weight_patch[30] = ele[10] + ele[8];
    product_weight_patch[31] = ele[10] + ele[11] + ele[8] + ele[9];
    product_weight_patch[32] = ele[11] + ele[9];
    product_weight_patch[33] = ele[10];
    product_weight_patch[34] = ele[10] + ele[11];
    product_weight_patch[35] = ele[11];


    for(int i = 0; i < 36; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (36, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x2x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[36] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[9 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[9 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[36] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[18] + input_patch[19] - input_patch[1] + input_patch[21] - input_patch[22] - input_patch[3] + input_patch[4];
    trans_input_patch[1] = -input_patch[19] + input_patch[1] + input_patch[22] - input_patch[4];
    trans_input_patch[2] = input_patch[19] - input_patch[1] - input_patch[20] - input_patch[22] + input_patch[23] + input_patch[2] + input_patch[4] - input_patch[5];
    trans_input_patch[3] = -input_patch[21] + input_patch[22] + input_patch[3] - input_patch[4];
    trans_input_patch[4] = -input_patch[22] + input_patch[4];
    trans_input_patch[5] = input_patch[22] - input_patch[23] - input_patch[4] + input_patch[5];
    trans_input_patch[6] = input_patch[21] - input_patch[22] - input_patch[24] + input_patch[25] - input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[7] = input_patch[22] - input_patch[25] - input_patch[4] + input_patch[7];
    trans_input_patch[8] = -input_patch[22] + input_patch[23] + input_patch[25] - input_patch[26] + input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[9] = -input_patch[10] - input_patch[12] + input_patch[13] + input_patch[18] - input_patch[19] - input_patch[21] + input_patch[22] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[13] + input_patch[19] - input_patch[22];
    trans_input_patch[11] = -input_patch[10] + input_patch[11] + input_patch[13] - input_patch[14] - input_patch[19] + input_patch[20] + input_patch[22] - input_patch[23];
    trans_input_patch[12] = input_patch[12] - input_patch[13] + input_patch[21] - input_patch[22];
    trans_input_patch[13] = input_patch[13] + input_patch[22];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] - input_patch[22] + input_patch[23];
    trans_input_patch[15] = -input_patch[12] + input_patch[13] + input_patch[15] - input_patch[16] - input_patch[21] + input_patch[22] + input_patch[24] - input_patch[25];
    trans_input_patch[16] = -input_patch[13] + input_patch[16] - input_patch[22] + input_patch[25];
    trans_input_patch[17] = input_patch[13] - input_patch[14] - input_patch[16] + input_patch[17] + input_patch[22] - input_patch[23] - input_patch[25] + input_patch[26];
    trans_input_patch[18] = input_patch[10] + input_patch[12] - input_patch[13] + input_patch[18] - input_patch[19] - input_patch[21] + input_patch[22] - input_patch[9];
    trans_input_patch[19] = -input_patch[10] + input_patch[13] + input_patch[19] - input_patch[22];
    trans_input_patch[20] = input_patch[10] - input_patch[11] - input_patch[13] + input_patch[14] - input_patch[19] + input_patch[20] + input_patch[22] - input_patch[23];
    trans_input_patch[21] = -input_patch[12] + input_patch[13] + input_patch[21] - input_patch[22];
    trans_input_patch[22] = -input_patch[13] + input_patch[22];
    trans_input_patch[23] = input_patch[13] - input_patch[14] - input_patch[22] + input_patch[23];
    trans_input_patch[24] = input_patch[12] - input_patch[13] - input_patch[15] + input_patch[16] - input_patch[21] + input_patch[22] + input_patch[24] - input_patch[25];
    trans_input_patch[25] = input_patch[13] - input_patch[16] - input_patch[22] + input_patch[25];
    trans_input_patch[26] = -input_patch[13] + input_patch[14] + input_patch[16] - input_patch[17] + input_patch[22] - input_patch[23] - input_patch[25] + input_patch[26];
    trans_input_patch[27] = -input_patch[10] - input_patch[12] + input_patch[13] - input_patch[27] + input_patch[28] + input_patch[30] - input_patch[31] + input_patch[9];
    trans_input_patch[28] = input_patch[10] - input_patch[13] - input_patch[28] + input_patch[31];
    trans_input_patch[29] = -input_patch[10] + input_patch[11] + input_patch[13] - input_patch[14] + input_patch[28] - input_patch[29] - input_patch[31] + input_patch[32];
    trans_input_patch[30] = input_patch[12] - input_patch[13] - input_patch[30] + input_patch[31];
    trans_input_patch[31] = input_patch[13] - input_patch[31];
    trans_input_patch[32] = -input_patch[13] + input_patch[14] + input_patch[31] - input_patch[32];
    trans_input_patch[33] = -input_patch[12] + input_patch[13] + input_patch[15] - input_patch[16] + input_patch[30] - input_patch[31] - input_patch[33] + input_patch[34];
    trans_input_patch[34] = -input_patch[13] + input_patch[16] + input_patch[31] - input_patch[34];
    trans_input_patch[35] = input_patch[13] - input_patch[14] - input_patch[16] + input_patch[17] - input_patch[31] + input_patch[32] + input_patch[34] - input_patch[35];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 36; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (36, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x2x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[36] = {T(0)};

    for(int i = 0; i < 36; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[12] + product_patch[13] + product_patch[18] + product_patch[19] + product_patch[1] + product_patch[21] + product_patch[22] + product_patch[3] + product_patch[4] + product_patch[9];
    output_patch[1] = product_patch[10] + product_patch[11] + product_patch[13] + product_patch[14] + product_patch[19] + product_patch[1] + product_patch[20] + product_patch[22] + product_patch[23] + product_patch[2] + product_patch[4] + product_patch[5];
    output_patch[2] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[21] + product_patch[22] + product_patch[24] + product_patch[25] + product_patch[3] + product_patch[4] + product_patch[6] + product_patch[7];
    output_patch[3] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[22] + product_patch[23] + product_patch[25] + product_patch[26] + product_patch[4] + product_patch[5] + product_patch[7] + product_patch[8];
    output_patch[4] = product_patch[10] + product_patch[12] + product_patch[13] - product_patch[18] - product_patch[19] - product_patch[21] - product_patch[22] - product_patch[27] - product_patch[28] - product_patch[30] - product_patch[31] + product_patch[9];
    output_patch[5] = product_patch[10] + product_patch[11] + product_patch[13] + product_patch[14] - product_patch[19] - product_patch[20] - product_patch[22] - product_patch[23] - product_patch[28] - product_patch[29] - product_patch[31] - product_patch[32];
    output_patch[6] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] - product_patch[21] - product_patch[22] - product_patch[24] - product_patch[25] - product_patch[30] - product_patch[31] - product_patch[33] - product_patch[34];
    output_patch[7] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] - product_patch[22] - product_patch[23] - product_patch[25] - product_patch[26] - product_patch[31] - product_patch[32] - product_patch[34] - product_patch[35];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (18, C, K)
// wino_weight = (48, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x2x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[18] = {T(0)};
    for(int i = 0; i < 18; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[48] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = ele[0] + ele[3];
    product_weight_patch[5] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5]);
    product_weight_patch[7] = ele[2] + ele[5];
    product_weight_patch[8] = ele[3];
    product_weight_patch[9] = T(1/2.)*(ele[3] + ele[4] + ele[5]);
    product_weight_patch[10] = T(1/2.)*(ele[3] - ele[4] + ele[5]);
    product_weight_patch[11] = ele[5];
    product_weight_patch[12] = T(1/2.)*(ele[0] + ele[12] + ele[6]);
    product_weight_patch[13] = T(1/4.)*(ele[0] + ele[12] + ele[13] + ele[14] + ele[1] + ele[2] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[14] = T(1/4.)*(ele[0] + ele[12] - ele[13] + ele[14] - ele[1] + ele[2] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[15] = T(1/2.)*(ele[14] + ele[2] + ele[8]);
    product_weight_patch[16] = T(1/2.)*(ele[0] + ele[12] + ele[15] + ele[3] + ele[6] + ele[9]);
    product_weight_patch[17] = T(1/4.)*(ele[0] + ele[10] + ele[11] + ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[18] = T(1/4.)*(ele[0] - ele[10] + ele[11] + ele[12] - ele[13] + ele[14] + ele[15] - ele[16] + ele[17] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[19] = T(1/2.)*(ele[11] + ele[14] + ele[17] + ele[2] + ele[5] + ele[8]);
    product_weight_patch[20] = T(1/2.)*(ele[15] + ele[3] + ele[9]);
    product_weight_patch[21] = T(1/4.)*(ele[10] + ele[11] + ele[15] + ele[16] + ele[17] + ele[3] + ele[4] + ele[5] + ele[9]);
    product_weight_patch[22] = T(1/4.)*(-ele[10] + ele[11] + ele[15] - ele[16] + ele[17] + ele[3] - ele[4] + ele[5] + ele[9]);
    product_weight_patch[23] = T(1/2.)*(ele[11] + ele[17] + ele[5]);
    product_weight_patch[24] = T(1/2.)*(ele[0] + ele[12] - ele[6]);
    product_weight_patch[25] = T(1/4.)*(ele[0] + ele[12] + ele[13] + ele[14] + ele[1] + ele[2] - ele[6] - ele[7] - ele[8]);
    product_weight_patch[26] = T(1/4.)*(ele[0] + ele[12] - ele[13] + ele[14] - ele[1] + ele[2] - ele[6] + ele[7] - ele[8]);
    product_weight_patch[27] = T(1/2.)*(ele[14] + ele[2] - ele[8]);
    product_weight_patch[28] = T(1/2.)*(ele[0] + ele[12] + ele[15] + ele[3] - ele[6] - ele[9]);
    product_weight_patch[29] = T(1/4.)*(ele[0] - ele[10] - ele[11] + ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] - ele[6] - ele[7] - ele[8] - ele[9]);
    product_weight_patch[30] = T(1/4.)*(ele[0] + ele[10] - ele[11] + ele[12] - ele[13] + ele[14] + ele[15] - ele[16] + ele[17] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] - ele[6] + ele[7] - ele[8] - ele[9]);
    product_weight_patch[31] = T(1/2.)*(-ele[11] + ele[14] + ele[17] + ele[2] + ele[5] - ele[8]);
    product_weight_patch[32] = T(1/2.)*(ele[15] + ele[3] - ele[9]);
    product_weight_patch[33] = T(1/4.)*(-ele[10] - ele[11] + ele[15] + ele[16] + ele[17] + ele[3] + ele[4] + ele[5] - ele[9]);
    product_weight_patch[34] = T(1/4.)*(ele[10] - ele[11] + ele[15] - ele[16] + ele[17] + ele[3] - ele[4] + ele[5] - ele[9]);
    product_weight_patch[35] = T(1/2.)*(-ele[11] + ele[17] + ele[5]);
    product_weight_patch[36] = ele[12];
    product_weight_patch[37] = T(1/2.)*(ele[12] + ele[13] + ele[14]);
    product_weight_patch[38] = T(1/2.)*(ele[12] - ele[13] + ele[14]);
    product_weight_patch[39] = ele[14];
    product_weight_patch[40] = ele[12] + ele[15];
    product_weight_patch[41] = T(1/2.)*(ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17]);
    product_weight_patch[42] = T(1/2.)*(ele[12] - ele[13] + ele[14] + ele[15] - ele[16] + ele[17]);
    product_weight_patch[43] = ele[14] + ele[17];
    product_weight_patch[44] = ele[15];
    product_weight_patch[45] = T(1/2.)*(ele[15] + ele[16] + ele[17]);
    product_weight_patch[46] = T(1/2.)*(ele[15] - ele[16] + ele[17]);
    product_weight_patch[47] = ele[17];


    for(int i = 0; i < 48; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (48, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x2x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 3) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[48] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[12 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[12 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[48] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[24] + input_patch[26] + input_patch[28] - input_patch[2] - input_patch[30] - input_patch[4] + input_patch[6];
    trans_input_patch[1] = input_patch[1] - input_patch[25] - input_patch[26] + input_patch[29] + input_patch[2] + input_patch[30] - input_patch[5] - input_patch[6];
    trans_input_patch[2] = -input_patch[1] + input_patch[25] - input_patch[26] - input_patch[29] + input_patch[2] + input_patch[30] + input_patch[5] - input_patch[6];
    trans_input_patch[3] = input_patch[1] - input_patch[25] + input_patch[27] + input_patch[29] - input_patch[31] - input_patch[3] - input_patch[5] + input_patch[7];
    trans_input_patch[4] = -input_patch[28] + input_patch[30] + input_patch[4] - input_patch[6];
    trans_input_patch[5] = -input_patch[29] - input_patch[30] + input_patch[5] + input_patch[6];
    trans_input_patch[6] = input_patch[29] - input_patch[30] - input_patch[5] + input_patch[6];
    trans_input_patch[7] = -input_patch[29] + input_patch[31] + input_patch[5] - input_patch[7];
    trans_input_patch[8] = -input_patch[10] + input_patch[28] - input_patch[30] - input_patch[32] + input_patch[34] - input_patch[4] + input_patch[6] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[29] + input_patch[30] - input_patch[33] - input_patch[34] - input_patch[5] - input_patch[6] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[29] + input_patch[30] + input_patch[33] - input_patch[34] + input_patch[5] - input_patch[6] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] + input_patch[29] - input_patch[31] - input_patch[33] + input_patch[35] - input_patch[5] + input_patch[7] + input_patch[9];
    trans_input_patch[12] = input_patch[12] - input_patch[14] - input_patch[16] + input_patch[18] + input_patch[24] - input_patch[26] - input_patch[28] + input_patch[30];
    trans_input_patch[13] = input_patch[13] + input_patch[14] - input_patch[17] - input_patch[18] + input_patch[25] + input_patch[26] - input_patch[29] - input_patch[30];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[17] - input_patch[18] - input_patch[25] + input_patch[26] + input_patch[29] - input_patch[30];
    trans_input_patch[15] = input_patch[13] - input_patch[15] - input_patch[17] + input_patch[19] + input_patch[25] - input_patch[27] - input_patch[29] + input_patch[31];
    trans_input_patch[16] = input_patch[16] - input_patch[18] + input_patch[28] - input_patch[30];
    trans_input_patch[17] = input_patch[17] + input_patch[18] + input_patch[29] + input_patch[30];
    trans_input_patch[18] = -input_patch[17] + input_patch[18] - input_patch[29] + input_patch[30];
    trans_input_patch[19] = input_patch[17] - input_patch[19] + input_patch[29] - input_patch[31];
    trans_input_patch[20] = -input_patch[16] + input_patch[18] + input_patch[20] - input_patch[22] - input_patch[28] + input_patch[30] + input_patch[32] - input_patch[34];
    trans_input_patch[21] = -input_patch[17] - input_patch[18] + input_patch[21] + input_patch[22] - input_patch[29] - input_patch[30] + input_patch[33] + input_patch[34];
    trans_input_patch[22] = input_patch[17] - input_patch[18] - input_patch[21] + input_patch[22] + input_patch[29] - input_patch[30] - input_patch[33] + input_patch[34];
    trans_input_patch[23] = -input_patch[17] + input_patch[19] + input_patch[21] - input_patch[23] - input_patch[29] + input_patch[31] + input_patch[33] - input_patch[35];
    trans_input_patch[24] = -input_patch[12] + input_patch[14] + input_patch[16] - input_patch[18] + input_patch[24] - input_patch[26] - input_patch[28] + input_patch[30];
    trans_input_patch[25] = -input_patch[13] - input_patch[14] + input_patch[17] + input_patch[18] + input_patch[25] + input_patch[26] - input_patch[29] - input_patch[30];
    trans_input_patch[26] = input_patch[13] - input_patch[14] - input_patch[17] + input_patch[18] - input_patch[25] + input_patch[26] + input_patch[29] - input_patch[30];
    trans_input_patch[27] = -input_patch[13] + input_patch[15] + input_patch[17] - input_patch[19] + input_patch[25] - input_patch[27] - input_patch[29] + input_patch[31];
    trans_input_patch[28] = -input_patch[16] + input_patch[18] + input_patch[28] - input_patch[30];
    trans_input_patch[29] = -input_patch[17] - input_patch[18] + input_patch[29] + input_patch[30];
    trans_input_patch[30] = input_patch[17] - input_patch[18] - input_patch[29] + input_patch[30];
    trans_input_patch[31] = -input_patch[17] + input_patch[19] + input_patch[29] - input_patch[31];
    trans_input_patch[32] = input_patch[16] - input_patch[18] - input_patch[20] + input_patch[22] - input_patch[28] + input_patch[30] + input_patch[32] - input_patch[34];
    trans_input_patch[33] = input_patch[17] + input_patch[18] - input_patch[21] - input_patch[22] - input_patch[29] - input_patch[30] + input_patch[33] + input_patch[34];
    trans_input_patch[34] = -input_patch[17] + input_patch[18] + input_patch[21] - input_patch[22] + input_patch[29] - input_patch[30] - input_patch[33] + input_patch[34];
    trans_input_patch[35] = input_patch[17] - input_patch[19] - input_patch[21] + input_patch[23] - input_patch[29] + input_patch[31] + input_patch[33] - input_patch[35];
    trans_input_patch[36] = input_patch[12] - input_patch[14] - input_patch[16] + input_patch[18] - input_patch[36] + input_patch[38] + input_patch[40] - input_patch[42];
    trans_input_patch[37] = input_patch[13] + input_patch[14] - input_patch[17] - input_patch[18] - input_patch[37] - input_patch[38] + input_patch[41] + input_patch[42];
    trans_input_patch[38] = -input_patch[13] + input_patch[14] + input_patch[17] - input_patch[18] + input_patch[37] - input_patch[38] - input_patch[41] + input_patch[42];
    trans_input_patch[39] = input_patch[13] - input_patch[15] - input_patch[17] + input_patch[19] - input_patch[37] + input_patch[39] + input_patch[41] - input_patch[43];
    trans_input_patch[40] = input_patch[16] - input_patch[18] - input_patch[40] + input_patch[42];
    trans_input_patch[41] = input_patch[17] + input_patch[18] - input_patch[41] - input_patch[42];
    trans_input_patch[42] = -input_patch[17] + input_patch[18] + input_patch[41] - input_patch[42];
    trans_input_patch[43] = input_patch[17] - input_patch[19] - input_patch[41] + input_patch[43];
    trans_input_patch[44] = -input_patch[16] + input_patch[18] + input_patch[20] - input_patch[22] + input_patch[40] - input_patch[42] - input_patch[44] + input_patch[46];
    trans_input_patch[45] = -input_patch[17] - input_patch[18] + input_patch[21] + input_patch[22] + input_patch[41] + input_patch[42] - input_patch[45] - input_patch[46];
    trans_input_patch[46] = input_patch[17] - input_patch[18] - input_patch[21] + input_patch[22] - input_patch[41] + input_patch[42] + input_patch[45] - input_patch[46];
    trans_input_patch[47] = -input_patch[17] + input_patch[19] + input_patch[21] - input_patch[23] + input_patch[41] - input_patch[43] - input_patch[45] + input_patch[47];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 48; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (48, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x2x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[48] = {T(0)};

    for(int i = 0; i < 48; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[12] + product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[1] + product_patch[24] + product_patch[25] + product_patch[26] + product_patch[28] + product_patch[29] + product_patch[2] + product_patch[30] + product_patch[4] + product_patch[5] + product_patch[6];
    output_patch[1] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[1] + product_patch[25] - product_patch[26] - product_patch[27] + product_patch[29] - product_patch[2] - product_patch[30] - product_patch[31] - product_patch[3] + product_patch[5] - product_patch[6] - product_patch[7];
    output_patch[2] = product_patch[10] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[28] + product_patch[29] + product_patch[30] + product_patch[32] + product_patch[33] + product_patch[34] + product_patch[4] + product_patch[5] + product_patch[6] + product_patch[8] + product_patch[9];
    output_patch[3] = -product_patch[10] - product_patch[11] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[29] - product_patch[30] - product_patch[31] + product_patch[33] - product_patch[34] - product_patch[35] + product_patch[5] - product_patch[6] - product_patch[7] + product_patch[9];
    output_patch[4] = product_patch[12] + product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[18] - product_patch[24] - product_patch[25] - product_patch[26] - product_patch[28] - product_patch[29] - product_patch[30] - product_patch[36] - product_patch[37] - product_patch[38] - product_patch[40] - product_patch[41] - product_patch[42];
    output_patch[5] = product_patch[13] - product_patch[14] - product_patch[15] + product_patch[17] - product_patch[18] - product_patch[19] - product_patch[25] + product_patch[26] + product_patch[27] - product_patch[29] + product_patch[30] + product_patch[31] - product_patch[37] + product_patch[38] + product_patch[39] - product_patch[41] + product_patch[42] + product_patch[43];
    output_patch[6] = product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22] - product_patch[28] - product_patch[29] - product_patch[30] - product_patch[32] - product_patch[33] - product_patch[34] - product_patch[40] - product_patch[41] - product_patch[42] - product_patch[44] - product_patch[45] - product_patch[46];
    output_patch[7] = product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23] - product_patch[29] + product_patch[30] + product_patch[31] - product_patch[33] + product_patch[34] + product_patch[35] - product_patch[41] + product_patch[42] + product_patch[43] - product_patch[45] + product_patch[46] + product_patch[47];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (9, C, K)
// wino_weight = (32, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x3x1(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[9] = {T(0)};
    for(int i = 0; i < 9; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[32] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0];
    product_weight_patch[2] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[3] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[4] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[5] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[6] = ele[2];
    product_weight_patch[7] = ele[2];
    product_weight_patch[8] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[9] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[10] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[11] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[12] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[13] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[14] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[15] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[16] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[17] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[18] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[19] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[20] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[21] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[22] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[23] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[24] = ele[6];
    product_weight_patch[25] = ele[6];
    product_weight_patch[26] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[27] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[28] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[29] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[30] = ele[8];
    product_weight_patch[31] = ele[8];


    for(int i = 0; i < 32; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (32, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x3x1(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 2) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[32] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 2; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[8 * i + 2 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[8 * i + 2 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[32] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[16] + input_patch[20] - input_patch[4];
    trans_input_patch[1] = -input_patch[17] + input_patch[1] + input_patch[21] - input_patch[5];
    trans_input_patch[2] = -input_patch[18] - input_patch[20] + input_patch[2] + input_patch[4];
    trans_input_patch[3] = -input_patch[19] - input_patch[21] + input_patch[3] + input_patch[5];
    trans_input_patch[4] = input_patch[18] - input_patch[20] - input_patch[2] + input_patch[4];
    trans_input_patch[5] = input_patch[19] - input_patch[21] - input_patch[3] + input_patch[5];
    trans_input_patch[6] = -input_patch[18] + input_patch[22] + input_patch[2] - input_patch[6];
    trans_input_patch[7] = -input_patch[19] + input_patch[23] + input_patch[3] - input_patch[7];
    trans_input_patch[8] = -input_patch[12] + input_patch[16] - input_patch[20] + input_patch[8];
    trans_input_patch[9] = -input_patch[13] + input_patch[17] - input_patch[21] + input_patch[9];
    trans_input_patch[10] = input_patch[10] + input_patch[12] + input_patch[18] + input_patch[20];
    trans_input_patch[11] = input_patch[11] + input_patch[13] + input_patch[19] + input_patch[21];
    trans_input_patch[12] = -input_patch[10] + input_patch[12] - input_patch[18] + input_patch[20];
    trans_input_patch[13] = -input_patch[11] + input_patch[13] - input_patch[19] + input_patch[21];
    trans_input_patch[14] = input_patch[10] - input_patch[14] + input_patch[18] - input_patch[22];
    trans_input_patch[15] = input_patch[11] - input_patch[15] + input_patch[19] - input_patch[23];
    trans_input_patch[16] = input_patch[12] + input_patch[16] - input_patch[20] - input_patch[8];
    trans_input_patch[17] = input_patch[13] + input_patch[17] - input_patch[21] - input_patch[9];
    trans_input_patch[18] = -input_patch[10] - input_patch[12] + input_patch[18] + input_patch[20];
    trans_input_patch[19] = -input_patch[11] - input_patch[13] + input_patch[19] + input_patch[21];
    trans_input_patch[20] = input_patch[10] - input_patch[12] - input_patch[18] + input_patch[20];
    trans_input_patch[21] = input_patch[11] - input_patch[13] - input_patch[19] + input_patch[21];
    trans_input_patch[22] = -input_patch[10] + input_patch[14] + input_patch[18] - input_patch[22];
    trans_input_patch[23] = -input_patch[11] + input_patch[15] + input_patch[19] - input_patch[23];
    trans_input_patch[24] = -input_patch[12] - input_patch[24] + input_patch[28] + input_patch[8];
    trans_input_patch[25] = -input_patch[13] - input_patch[25] + input_patch[29] + input_patch[9];
    trans_input_patch[26] = input_patch[10] + input_patch[12] - input_patch[26] - input_patch[28];
    trans_input_patch[27] = input_patch[11] + input_patch[13] - input_patch[27] - input_patch[29];
    trans_input_patch[28] = -input_patch[10] + input_patch[12] + input_patch[26] - input_patch[28];
    trans_input_patch[29] = -input_patch[11] + input_patch[13] + input_patch[27] - input_patch[29];
    trans_input_patch[30] = input_patch[10] - input_patch[14] - input_patch[26] + input_patch[30];
    trans_input_patch[31] = input_patch[11] - input_patch[15] - input_patch[27] + input_patch[31];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 32; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (32, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x3x1(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[32] = {T(0)};

    for(int i = 0; i < 32; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[12] + product_patch[16] + product_patch[18] + product_patch[20] + product_patch[2] + product_patch[4] + product_patch[8];
    output_patch[1] = product_patch[11] + product_patch[13] + product_patch[17] + product_patch[19] + product_patch[1] + product_patch[21] + product_patch[3] + product_patch[5] + product_patch[9];
    output_patch[2] = product_patch[10] - product_patch[12] - product_patch[14] + product_patch[18] - product_patch[20] - product_patch[22] + product_patch[2] - product_patch[4] - product_patch[6];
    output_patch[3] = product_patch[11] - product_patch[13] - product_patch[15] + product_patch[19] - product_patch[21] - product_patch[23] + product_patch[3] - product_patch[5] - product_patch[7];
    output_patch[4] = product_patch[10] + product_patch[12] - product_patch[16] - product_patch[18] - product_patch[20] - product_patch[24] - product_patch[26] - product_patch[28] + product_patch[8];
    output_patch[5] = product_patch[11] + product_patch[13] - product_patch[17] - product_patch[19] - product_patch[21] - product_patch[25] - product_patch[27] - product_patch[29] + product_patch[9];
    output_patch[6] = product_patch[10] - product_patch[12] - product_patch[14] - product_patch[18] + product_patch[20] + product_patch[22] - product_patch[26] + product_patch[28] + product_patch[30];
    output_patch[7] = product_patch[11] - product_patch[13] - product_patch[15] - product_patch[19] + product_patch[21] + product_patch[23] - product_patch[27] + product_patch[29] + product_patch[31];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (18, C, K)
// wino_weight = (48, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x3x2(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[18] = {T(0)};
    for(int i = 0; i < 18; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[48] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = ele[0] + ele[1];
    product_weight_patch[2] = ele[1];
    product_weight_patch[3] = T(1/2.)*(ele[0] + ele[2] + ele[4]);
    product_weight_patch[4] = T(1/2.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5]);
    product_weight_patch[5] = T(1/2.)*(ele[1] + ele[3] + ele[5]);
    product_weight_patch[6] = T(1/2.)*(ele[0] - ele[2] + ele[4]);
    product_weight_patch[7] = T(1/2.)*(ele[0] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5]);
    product_weight_patch[8] = T(1/2.)*(ele[1] - ele[3] + ele[5]);
    product_weight_patch[9] = ele[4];
    product_weight_patch[10] = ele[4] + ele[5];
    product_weight_patch[11] = ele[5];
    product_weight_patch[12] = T(1/2.)*(ele[0] + ele[12] + ele[6]);
    product_weight_patch[13] = T(1/2.)*(ele[0] + ele[12] + ele[13] + ele[1] + ele[6] + ele[7]);
    product_weight_patch[14] = T(1/2.)*(ele[13] + ele[1] + ele[7]);
    product_weight_patch[15] = T(1/4.)*(ele[0] + ele[10] + ele[12] + ele[14] + ele[16] + ele[2] + ele[4] + ele[6] + ele[8]);
    product_weight_patch[16] = T(1/4.)*(ele[0] + ele[10] + ele[11] + ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[17] = T(1/4.)*(ele[11] + ele[13] + ele[15] + ele[17] + ele[1] + ele[3] + ele[5] + ele[7] + ele[9]);
    product_weight_patch[18] = T(1/4.)*(ele[0] + ele[10] + ele[12] - ele[14] + ele[16] - ele[2] + ele[4] + ele[6] - ele[8]);
    product_weight_patch[19] = T(1/4.)*(ele[0] + ele[10] + ele[11] + ele[12] + ele[13] - ele[14] - ele[15] + ele[16] + ele[17] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5] + ele[6] + ele[7] - ele[8] - ele[9]);
    product_weight_patch[20] = T(1/4.)*(ele[11] + ele[13] - ele[15] + ele[17] + ele[1] - ele[3] + ele[5] + ele[7] - ele[9]);
    product_weight_patch[21] = T(1/2.)*(ele[10] + ele[16] + ele[4]);
    product_weight_patch[22] = T(1/2.)*(ele[10] + ele[11] + ele[16] + ele[17] + ele[4] + ele[5]);
    product_weight_patch[23] = T(1/2.)*(ele[11] + ele[17] + ele[5]);
    product_weight_patch[24] = T(1/2.)*(ele[0] + ele[12] - ele[6]);
    product_weight_patch[25] = T(1/2.)*(ele[0] + ele[12] + ele[13] + ele[1] - ele[6] - ele[7]);
    product_weight_patch[26] = T(1/2.)*(ele[13] + ele[1] - ele[7]);
    product_weight_patch[27] = T(1/4.)*(ele[0] - ele[10] + ele[12] + ele[14] + ele[16] + ele[2] + ele[4] - ele[6] - ele[8]);
    product_weight_patch[28] = T(1/4.)*(ele[0] - ele[10] - ele[11] + ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] - ele[6] - ele[7] - ele[8] - ele[9]);
    product_weight_patch[29] = T(1/4.)*(-ele[11] + ele[13] + ele[15] + ele[17] + ele[1] + ele[3] + ele[5] - ele[7] - ele[9]);
    product_weight_patch[30] = T(1/4.)*(ele[0] - ele[10] + ele[12] - ele[14] + ele[16] - ele[2] + ele[4] - ele[6] + ele[8]);
    product_weight_patch[31] = T(1/4.)*(ele[0] - ele[10] - ele[11] + ele[12] + ele[13] - ele[14] - ele[15] + ele[16] + ele[17] + ele[1] - ele[2] - ele[3] + ele[4] + ele[5] - ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[32] = T(1/4.)*(-ele[11] + ele[13] - ele[15] + ele[17] + ele[1] - ele[3] + ele[5] - ele[7] + ele[9]);
    product_weight_patch[33] = T(1/2.)*(-ele[10] + ele[16] + ele[4]);
    product_weight_patch[34] = T(1/2.)*(-ele[10] - ele[11] + ele[16] + ele[17] + ele[4] + ele[5]);
    product_weight_patch[35] = T(1/2.)*(-ele[11] + ele[17] + ele[5]);
    product_weight_patch[36] = ele[12];
    product_weight_patch[37] = ele[12] + ele[13];
    product_weight_patch[38] = ele[13];
    product_weight_patch[39] = T(1/2.)*(ele[12] + ele[14] + ele[16]);
    product_weight_patch[40] = T(1/2.)*(ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17]);
    product_weight_patch[41] = T(1/2.)*(ele[13] + ele[15] + ele[17]);
    product_weight_patch[42] = T(1/2.)*(ele[12] - ele[14] + ele[16]);
    product_weight_patch[43] = T(1/2.)*(ele[12] + ele[13] - ele[14] - ele[15] + ele[16] + ele[17]);
    product_weight_patch[44] = T(1/2.)*(ele[13] - ele[15] + ele[17]);
    product_weight_patch[45] = ele[16];
    product_weight_patch[46] = ele[16] + ele[17];
    product_weight_patch[47] = ele[17];


    for(int i = 0; i < 48; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (48, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x3x2(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 3) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[48] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 3; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[12 * i + 3 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[12 * i + 3 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[48] = {T(0)};


    trans_input_patch[0] = input_patch[0] - input_patch[1] - input_patch[24] + input_patch[25] + input_patch[30] - input_patch[31] - input_patch[6] + input_patch[7];
    trans_input_patch[1] = input_patch[1] - input_patch[25] + input_patch[31] - input_patch[7];
    trans_input_patch[2] = -input_patch[1] + input_patch[25] - input_patch[26] + input_patch[2] - input_patch[31] + input_patch[32] + input_patch[7] - input_patch[8];
    trans_input_patch[3] = -input_patch[27] + input_patch[28] - input_patch[30] + input_patch[31] + input_patch[3] - input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[4] = -input_patch[28] - input_patch[31] + input_patch[4] + input_patch[7];
    trans_input_patch[5] = input_patch[28] - input_patch[29] + input_patch[31] - input_patch[32] - input_patch[4] + input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[6] = input_patch[27] - input_patch[28] - input_patch[30] + input_patch[31] - input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
    trans_input_patch[7] = input_patch[28] - input_patch[31] - input_patch[4] + input_patch[7];
    trans_input_patch[8] = -input_patch[28] + input_patch[29] + input_patch[31] - input_patch[32] + input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
    trans_input_patch[9] = input_patch[10] - input_patch[27] + input_patch[28] + input_patch[33] - input_patch[34] + input_patch[3] - input_patch[4] - input_patch[9];
    trans_input_patch[10] = -input_patch[10] - input_patch[28] + input_patch[34] + input_patch[4];
    trans_input_patch[11] = input_patch[10] - input_patch[11] + input_patch[28] - input_patch[29] - input_patch[34] + input_patch[35] - input_patch[4] + input_patch[5];
    trans_input_patch[12] = input_patch[12] - input_patch[13] - input_patch[18] + input_patch[19] + input_patch[24] - input_patch[25] - input_patch[30] + input_patch[31];
    trans_input_patch[13] = input_patch[13] - input_patch[19] + input_patch[25] - input_patch[31];
    trans_input_patch[14] = -input_patch[13] + input_patch[14] + input_patch[19] - input_patch[20] - input_patch[25] + input_patch[26] + input_patch[31] - input_patch[32];
    trans_input_patch[15] = input_patch[15] - input_patch[16] + input_patch[18] - input_patch[19] + input_patch[27] - input_patch[28] + input_patch[30] - input_patch[31];
    trans_input_patch[16] = input_patch[16] + input_patch[19] + input_patch[28] + input_patch[31];
    trans_input_patch[17] = -input_patch[16] + input_patch[17] - input_patch[19] + input_patch[20] - input_patch[28] + input_patch[29] - input_patch[31] + input_patch[32];
    trans_input_patch[18] = -input_patch[15] + input_patch[16] + input_patch[18] - input_patch[19] - input_patch[27] + input_patch[28] + input_patch[30] - input_patch[31];
    trans_input_patch[19] = -input_patch[16] + input_patch[19] - input_patch[28] + input_patch[31];
    trans_input_patch[20] = input_patch[16] - input_patch[17] - input_patch[19] + input_patch[20] + input_patch[28] - input_patch[29] - input_patch[31] + input_patch[32];
    trans_input_patch[21] = input_patch[15] - input_patch[16] - input_patch[21] + input_patch[22] + input_patch[27] - input_patch[28] - input_patch[33] + input_patch[34];
    trans_input_patch[22] = input_patch[16] - input_patch[22] + input_patch[28] - input_patch[34];
    trans_input_patch[23] = -input_patch[16] + input_patch[17] + input_patch[22] - input_patch[23] - input_patch[28] + input_patch[29] + input_patch[34] - input_patch[35];
    trans_input_patch[24] = -input_patch[12] + input_patch[13] + input_patch[18] - input_patch[19] + input_patch[24] - input_patch[25] - input_patch[30] + input_patch[31];
    trans_input_patch[25] = -input_patch[13] + input_patch[19] + input_patch[25] - input_patch[31];
    trans_input_patch[26] = input_patch[13] - input_patch[14] - input_patch[19] + input_patch[20] - input_patch[25] + input_patch[26] + input_patch[31] - input_patch[32];
    trans_input_patch[27] = -input_patch[15] + input_patch[16] - input_patch[18] + input_patch[19] + input_patch[27] - input_patch[28] + input_patch[30] - input_patch[31];
    trans_input_patch[28] = -input_patch[16] - input_patch[19] + input_patch[28] + input_patch[31];
    trans_input_patch[29] = input_patch[16] - input_patch[17] + input_patch[19] - input_patch[20] - input_patch[28] + input_patch[29] - input_patch[31] + input_patch[32];
    trans_input_patch[30] = input_patch[15] - input_patch[16] - input_patch[18] + input_patch[19] - input_patch[27] + input_patch[28] + input_patch[30] - input_patch[31];
    trans_input_patch[31] = input_patch[16] - input_patch[19] - input_patch[28] + input_patch[31];
    trans_input_patch[32] = -input_patch[16] + input_patch[17] + input_patch[19] - input_patch[20] + input_patch[28] - input_patch[29] - input_patch[31] + input_patch[32];
    trans_input_patch[33] = -input_patch[15] + input_patch[16] + input_patch[21] - input_patch[22] + input_patch[27] - input_patch[28] - input_patch[33] + input_patch[34];
    trans_input_patch[34] = -input_patch[16] + input_patch[22] + input_patch[28] - input_patch[34];
    trans_input_patch[35] = input_patch[16] - input_patch[17] - input_patch[22] + input_patch[23] - input_patch[28] + input_patch[29] + input_patch[34] - input_patch[35];
    trans_input_patch[36] = input_patch[12] - input_patch[13] - input_patch[18] + input_patch[19] - input_patch[36] + input_patch[37] + input_patch[42] - input_patch[43];
    trans_input_patch[37] = input_patch[13] - input_patch[19] - input_patch[37] + input_patch[43];
    trans_input_patch[38] = -input_patch[13] + input_patch[14] + input_patch[19] - input_patch[20] + input_patch[37] - input_patch[38] - input_patch[43] + input_patch[44];
    trans_input_patch[39] = input_patch[15] - input_patch[16] + input_patch[18] - input_patch[19] - input_patch[39] + input_patch[40] - input_patch[42] + input_patch[43];
    trans_input_patch[40] = input_patch[16] + input_patch[19] - input_patch[40] - input_patch[43];
    trans_input_patch[41] = -input_patch[16] + input_patch[17] - input_patch[19] + input_patch[20] + input_patch[40] - input_patch[41] + input_patch[43] - input_patch[44];
    trans_input_patch[42] = -input_patch[15] + input_patch[16] + input_patch[18] - input_patch[19] + input_patch[39] - input_patch[40] - input_patch[42] + input_patch[43];
    trans_input_patch[43] = -input_patch[16] + input_patch[19] + input_patch[40] - input_patch[43];
    trans_input_patch[44] = input_patch[16] - input_patch[17] - input_patch[19] + input_patch[20] - input_patch[40] + input_patch[41] + input_patch[43] - input_patch[44];
    trans_input_patch[45] = input_patch[15] - input_patch[16] - input_patch[21] + input_patch[22] - input_patch[39] + input_patch[40] + input_patch[45] - input_patch[46];
    trans_input_patch[46] = input_patch[16] - input_patch[22] - input_patch[40] + input_patch[46];
    trans_input_patch[47] = -input_patch[16] + input_patch[17] + input_patch[22] - input_patch[23] + input_patch[40] - input_patch[41] - input_patch[46] + input_patch[47];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 48; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (48, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x3x2(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[48] = {T(0)};

    for(int i = 0; i < 48; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[18] + product_patch[19] + product_patch[1] + product_patch[24] + product_patch[25] + product_patch[27] + product_patch[28] + product_patch[30] + product_patch[31] + product_patch[3] + product_patch[4] + product_patch[6] + product_patch[7];
    output_patch[1] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[19] + product_patch[1] + product_patch[20] + product_patch[25] + product_patch[26] + product_patch[28] + product_patch[29] + product_patch[2] + product_patch[31] + product_patch[32] + product_patch[4] + product_patch[5] + product_patch[7] + product_patch[8];
    output_patch[2] = -product_patch[10] + product_patch[15] + product_patch[16] - product_patch[18] - product_patch[19] - product_patch[21] - product_patch[22] + product_patch[27] + product_patch[28] - product_patch[30] - product_patch[31] - product_patch[33] - product_patch[34] + product_patch[3] + product_patch[4] - product_patch[6] - product_patch[7] - product_patch[9];
    output_patch[3] = -product_patch[10] - product_patch[11] + product_patch[16] + product_patch[17] - product_patch[19] - product_patch[20] - product_patch[22] - product_patch[23] + product_patch[28] + product_patch[29] - product_patch[31] - product_patch[32] - product_patch[34] - product_patch[35] + product_patch[4] + product_patch[5] - product_patch[7] - product_patch[8];
    output_patch[4] = product_patch[12] + product_patch[13] + product_patch[15] + product_patch[16] + product_patch[18] + product_patch[19] - product_patch[24] - product_patch[25] - product_patch[27] - product_patch[28] - product_patch[30] - product_patch[31] - product_patch[36] - product_patch[37] - product_patch[39] - product_patch[40] - product_patch[42] - product_patch[43];
    output_patch[5] = product_patch[13] + product_patch[14] + product_patch[16] + product_patch[17] + product_patch[19] + product_patch[20] - product_patch[25] - product_patch[26] - product_patch[28] - product_patch[29] - product_patch[31] - product_patch[32] - product_patch[37] - product_patch[38] - product_patch[40] - product_patch[41] - product_patch[43] - product_patch[44];
    output_patch[6] = product_patch[15] + product_patch[16] - product_patch[18] - product_patch[19] - product_patch[21] - product_patch[22] - product_patch[27] - product_patch[28] + product_patch[30] + product_patch[31] + product_patch[33] + product_patch[34] - product_patch[39] - product_patch[40] + product_patch[42] + product_patch[43] + product_patch[45] + product_patch[46];
    output_patch[7] = product_patch[16] + product_patch[17] - product_patch[19] - product_patch[20] - product_patch[22] - product_patch[23] - product_patch[28] - product_patch[29] + product_patch[31] + product_patch[32] + product_patch[34] + product_patch[35] - product_patch[40] - product_patch[41] + product_patch[43] + product_patch[44] + product_patch[46] + product_patch[47];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}



// dim3 threadsPerBlock(K)
// dim3 numBlocks(1, 1, C)
// norm_weight = (27, C, K)
// wino_weight = (64, C, K)
template <typename T>
__global__ void wNorm2WinoTransform3x3x3(const T *norm_weight, T* wino_weight, int C, int K)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;


    T ele[27] = {T(0)};
    for(int i = 0; i < 27; i++) {
      ele[i] = norm_weight [i * C * K + bx * K + tx];
    }

    T product_weight_patch[64] = {T(0)};


    product_weight_patch[0] = ele[0];
    product_weight_patch[1] = T(1/2.)*(ele[0] + ele[1] + ele[2]);
    product_weight_patch[2] = T(1/2.)*(ele[0] - ele[1] + ele[2]);
    product_weight_patch[3] = ele[2];
    product_weight_patch[4] = T(1/2.)*(ele[0] + ele[3] + ele[6]);
    product_weight_patch[5] = T(1/4.)*(ele[0] + ele[1] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[6] = T(1/4.)*(ele[0] - ele[1] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[7] = T(1/2.)*(ele[2] + ele[5] + ele[8]);
    product_weight_patch[8] = T(1/2.)*(ele[0] - ele[3] + ele[6]);
    product_weight_patch[9] = T(1/4.)*(ele[0] + ele[1] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[10] = T(1/4.)*(ele[0] - ele[1] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[11] = T(1/2.)*(ele[2] - ele[5] + ele[8]);
    product_weight_patch[12] = ele[6];
    product_weight_patch[13] = T(1/2.)*(ele[6] + ele[7] + ele[8]);
    product_weight_patch[14] = T(1/2.)*(ele[6] - ele[7] + ele[8]);
    product_weight_patch[15] = ele[8];
    product_weight_patch[16] = T(1/2.)*(ele[0] + ele[18] + ele[9]);
    product_weight_patch[17] = T(1/4.)*(ele[0] + ele[10] + ele[11] + ele[18] + ele[19] + ele[1] + ele[20] + ele[2] + ele[9]);
    product_weight_patch[18] = T(1/4.)*(ele[0] - ele[10] + ele[11] + ele[18] - ele[19] - ele[1] + ele[20] + ele[2] + ele[9]);
    product_weight_patch[19] = T(1/2.)*(ele[11] + ele[20] + ele[2]);
    product_weight_patch[20] = T(1/4.)*(ele[0] + ele[12] + ele[15] + ele[18] + ele[21] + ele[24] + ele[3] + ele[6] + ele[9]);
    product_weight_patch[21] = T(1/8.)*(ele[0] + ele[10] + ele[11] + ele[12] + ele[13] + ele[14] + ele[15] + ele[16] + ele[17] + ele[18] + ele[19] + ele[1] + ele[20] + ele[21] + ele[22] + ele[23] + ele[24] + ele[25] + ele[26] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[22] = T(1/8.)*(ele[0] - ele[10] + ele[11] + ele[12] - ele[13] + ele[14] + ele[15] - ele[16] + ele[17] + ele[18] - ele[19] - ele[1] + ele[20] + ele[21] - ele[22] + ele[23] + ele[24] - ele[25] + ele[26] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[23] = T(1/4.)*(ele[11] + ele[14] + ele[17] + ele[20] + ele[23] + ele[26] + ele[2] + ele[5] + ele[8]);
    product_weight_patch[24] = T(1/4.)*(ele[0] - ele[12] + ele[15] + ele[18] - ele[21] + ele[24] - ele[3] + ele[6] + ele[9]);
    product_weight_patch[25] = T(1/8.)*(ele[0] + ele[10] + ele[11] - ele[12] - ele[13] - ele[14] + ele[15] + ele[16] + ele[17] + ele[18] + ele[19] + ele[1] + ele[20] - ele[21] - ele[22] - ele[23] + ele[24] + ele[25] + ele[26] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8] + ele[9]);
    product_weight_patch[26] = T(1/8.)*(ele[0] - ele[10] + ele[11] - ele[12] + ele[13] - ele[14] + ele[15] - ele[16] + ele[17] + ele[18] - ele[19] - ele[1] + ele[20] - ele[21] + ele[22] - ele[23] + ele[24] - ele[25] + ele[26] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8] + ele[9]);
    product_weight_patch[27] = T(1/4.)*(ele[11] - ele[14] + ele[17] + ele[20] - ele[23] + ele[26] + ele[2] - ele[5] + ele[8]);
    product_weight_patch[28] = T(1/2.)*(ele[15] + ele[24] + ele[6]);
    product_weight_patch[29] = T(1/4.)*(ele[15] + ele[16] + ele[17] + ele[24] + ele[25] + ele[26] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[30] = T(1/4.)*(ele[15] - ele[16] + ele[17] + ele[24] - ele[25] + ele[26] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[31] = T(1/2.)*(ele[17] + ele[26] + ele[8]);
    product_weight_patch[32] = T(1/2.)*(ele[0] + ele[18] - ele[9]);
    product_weight_patch[33] = T(1/4.)*(ele[0] - ele[10] - ele[11] + ele[18] + ele[19] + ele[1] + ele[20] + ele[2] - ele[9]);
    product_weight_patch[34] = T(1/4.)*(ele[0] + ele[10] - ele[11] + ele[18] - ele[19] - ele[1] + ele[20] + ele[2] - ele[9]);
    product_weight_patch[35] = T(1/2.)*(-ele[11] + ele[20] + ele[2]);
    product_weight_patch[36] = T(1/4.)*(ele[0] - ele[12] - ele[15] + ele[18] + ele[21] + ele[24] + ele[3] + ele[6] - ele[9]);
    product_weight_patch[37] = T(1/8.)*(ele[0] - ele[10] - ele[11] - ele[12] - ele[13] - ele[14] - ele[15] - ele[16] - ele[17] + ele[18] + ele[19] + ele[1] + ele[20] + ele[21] + ele[22] + ele[23] + ele[24] + ele[25] + ele[26] + ele[2] + ele[3] + ele[4] + ele[5] + ele[6] + ele[7] + ele[8] - ele[9]);
    product_weight_patch[38] = T(1/8.)*(ele[0] + ele[10] - ele[11] - ele[12] + ele[13] - ele[14] - ele[15] + ele[16] - ele[17] + ele[18] - ele[19] - ele[1] + ele[20] + ele[21] - ele[22] + ele[23] + ele[24] - ele[25] + ele[26] + ele[2] + ele[3] - ele[4] + ele[5] + ele[6] - ele[7] + ele[8] - ele[9]);
    product_weight_patch[39] = T(1/4.)*(-ele[11] - ele[14] - ele[17] + ele[20] + ele[23] + ele[26] + ele[2] + ele[5] + ele[8]);
    product_weight_patch[40] = T(1/4.)*(ele[0] + ele[12] - ele[15] + ele[18] - ele[21] + ele[24] - ele[3] + ele[6] - ele[9]);
    product_weight_patch[41] = T(1/8.)*(ele[0] - ele[10] - ele[11] + ele[12] + ele[13] + ele[14] - ele[15] - ele[16] - ele[17] + ele[18] + ele[19] + ele[1] + ele[20] - ele[21] - ele[22] - ele[23] + ele[24] + ele[25] + ele[26] + ele[2] - ele[3] - ele[4] - ele[5] + ele[6] + ele[7] + ele[8] - ele[9]);
    product_weight_patch[42] = T(1/8.)*(ele[0] + ele[10] - ele[11] + ele[12] - ele[13] + ele[14] - ele[15] + ele[16] - ele[17] + ele[18] - ele[19] - ele[1] + ele[20] - ele[21] + ele[22] - ele[23] + ele[24] - ele[25] + ele[26] + ele[2] - ele[3] + ele[4] - ele[5] + ele[6] - ele[7] + ele[8] - ele[9]);
    product_weight_patch[43] = T(1/4.)*(-ele[11] + ele[14] - ele[17] + ele[20] - ele[23] + ele[26] + ele[2] - ele[5] + ele[8]);
    product_weight_patch[44] = T(1/2.)*(-ele[15] + ele[24] + ele[6]);
    product_weight_patch[45] = T(1/4.)*(-ele[15] - ele[16] - ele[17] + ele[24] + ele[25] + ele[26] + ele[6] + ele[7] + ele[8]);
    product_weight_patch[46] = T(1/4.)*(-ele[15] + ele[16] - ele[17] + ele[24] - ele[25] + ele[26] + ele[6] - ele[7] + ele[8]);
    product_weight_patch[47] = T(1/2.)*(-ele[17] + ele[26] + ele[8]);
    product_weight_patch[48] = ele[18];
    product_weight_patch[49] = T(1/2.)*(ele[18] + ele[19] + ele[20]);
    product_weight_patch[50] = T(1/2.)*(ele[18] - ele[19] + ele[20]);
    product_weight_patch[51] = ele[20];
    product_weight_patch[52] = T(1/2.)*(ele[18] + ele[21] + ele[24]);
    product_weight_patch[53] = T(1/4.)*(ele[18] + ele[19] + ele[20] + ele[21] + ele[22] + ele[23] + ele[24] + ele[25] + ele[26]);
    product_weight_patch[54] = T(1/4.)*(ele[18] - ele[19] + ele[20] + ele[21] - ele[22] + ele[23] + ele[24] - ele[25] + ele[26]);
    product_weight_patch[55] = T(1/2.)*(ele[20] + ele[23] + ele[26]);
    product_weight_patch[56] = T(1/2.)*(ele[18] - ele[21] + ele[24]);
    product_weight_patch[57] = T(1/4.)*(ele[18] + ele[19] + ele[20] - ele[21] - ele[22] - ele[23] + ele[24] + ele[25] + ele[26]);
    product_weight_patch[58] = T(1/4.)*(ele[18] - ele[19] + ele[20] - ele[21] + ele[22] - ele[23] + ele[24] - ele[25] + ele[26]);
    product_weight_patch[59] = T(1/2.)*(ele[20] - ele[23] + ele[26]);
    product_weight_patch[60] = ele[24];
    product_weight_patch[61] = T(1/2.)*(ele[24] + ele[25] + ele[26]);
    product_weight_patch[62] = T(1/2.)*(ele[24] - ele[25] + ele[26]);
    product_weight_patch[63] = ele[26];


    for(int i = 0; i < 64; i++) {
      wino_weight [i * C * K + bx * K + tx] = T(product_weight_patch[i]);
    }

}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)

// I = (Batch, D, H, W, C)
// O = (64, Batch, nD, nH, nW, C)
template <typename T>
__global__ void inputNorm2WinoTransform3x3x3(const T *norm_input, T *wino_input, int B, int D, int H, int W, int C, int pad_D, int pad_h, int pad_w) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //c


    int nD = (D + 1 + 2 * pad_D - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;
    int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;


    int f_b = bz;
    int zBase = 2 * by - pad_D;
    int xBase = 2 * bx - pad_w;
    int yBase = 2 * ty - pad_h;


    T input_patch[64] = {T(0)};


    int f_x, f_y, f_z;
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 4; j++) {
        for(int k = 0; k < 4; k++) {
          f_z = zBase + i; f_y = yBase + j; f_x = xBase + k;
          if((f_z > -1) && (f_z < D) && (f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) {
            input_patch[16 * i + 4 * j + k] = norm_input[ (((f_b * D + f_z) * H + f_y) * W + f_x) * C + tx];
          } else {
            input_patch[16 * i + 4 * j + k] = T(0);
          }
        }
      }
    }

    T trans_input_patch[64] = {T(0)};


    trans_input_patch[0] = input_patch[0] + input_patch[10] - input_patch[2] - input_patch[32] + input_patch[34] + input_patch[40] - input_patch[42] - input_patch[8];
    trans_input_patch[1] = -input_patch[10] + input_patch[1] + input_patch[2] - input_patch[33] - input_patch[34] + input_patch[41] + input_patch[42] - input_patch[9];
    trans_input_patch[2] = -input_patch[10] - input_patch[1] + input_patch[2] + input_patch[33] - input_patch[34] - input_patch[41] + input_patch[42] + input_patch[9];
    trans_input_patch[3] = input_patch[11] + input_patch[1] - input_patch[33] + input_patch[35] - input_patch[3] + input_patch[41] - input_patch[43] - input_patch[9];
    trans_input_patch[4] = -input_patch[10] - input_patch[36] + input_patch[38] - input_patch[40] + input_patch[42] + input_patch[4] - input_patch[6] + input_patch[8];
    trans_input_patch[5] = input_patch[10] - input_patch[37] - input_patch[38] - input_patch[41] - input_patch[42] + input_patch[5] + input_patch[6] + input_patch[9];
    trans_input_patch[6] = input_patch[10] + input_patch[37] - input_patch[38] + input_patch[41] - input_patch[42] - input_patch[5] + input_patch[6] - input_patch[9];
    trans_input_patch[7] = -input_patch[11] - input_patch[37] + input_patch[39] - input_patch[41] + input_patch[43] + input_patch[5] - input_patch[7] + input_patch[9];
    trans_input_patch[8] = -input_patch[10] + input_patch[36] - input_patch[38] - input_patch[40] + input_patch[42] - input_patch[4] + input_patch[6] + input_patch[8];
    trans_input_patch[9] = input_patch[10] + input_patch[37] + input_patch[38] - input_patch[41] - input_patch[42] - input_patch[5] - input_patch[6] + input_patch[9];
    trans_input_patch[10] = input_patch[10] - input_patch[37] + input_patch[38] + input_patch[41] - input_patch[42] + input_patch[5] - input_patch[6] - input_patch[9];
    trans_input_patch[11] = -input_patch[11] + input_patch[37] - input_patch[39] - input_patch[41] + input_patch[43] - input_patch[5] + input_patch[7] + input_patch[9];
    trans_input_patch[12] = -input_patch[12] + input_patch[14] - input_patch[36] + input_patch[38] + input_patch[44] - input_patch[46] + input_patch[4] - input_patch[6];
    trans_input_patch[13] = -input_patch[13] - input_patch[14] - input_patch[37] - input_patch[38] + input_patch[45] + input_patch[46] + input_patch[5] + input_patch[6];
    trans_input_patch[14] = input_patch[13] - input_patch[14] + input_patch[37] - input_patch[38] - input_patch[45] + input_patch[46] - input_patch[5] + input_patch[6];
    trans_input_patch[15] = -input_patch[13] + input_patch[15] - input_patch[37] + input_patch[39] + input_patch[45] - input_patch[47] + input_patch[5] - input_patch[7];
    trans_input_patch[16] = input_patch[16] - input_patch[18] - input_patch[24] + input_patch[26] + input_patch[32] - input_patch[34] - input_patch[40] + input_patch[42];
    trans_input_patch[17] = input_patch[17] + input_patch[18] - input_patch[25] - input_patch[26] + input_patch[33] + input_patch[34] - input_patch[41] - input_patch[42];
    trans_input_patch[18] = -input_patch[17] + input_patch[18] + input_patch[25] - input_patch[26] - input_patch[33] + input_patch[34] + input_patch[41] - input_patch[42];
    trans_input_patch[19] = input_patch[17] - input_patch[19] - input_patch[25] + input_patch[27] + input_patch[33] - input_patch[35] - input_patch[41] + input_patch[43];
    trans_input_patch[20] = input_patch[20] - input_patch[22] + input_patch[24] - input_patch[26] + input_patch[36] - input_patch[38] + input_patch[40] - input_patch[42];
    trans_input_patch[21] = input_patch[21] + input_patch[22] + input_patch[25] + input_patch[26] + input_patch[37] + input_patch[38] + input_patch[41] + input_patch[42];
    trans_input_patch[22] = -input_patch[21] + input_patch[22] - input_patch[25] + input_patch[26] - input_patch[37] + input_patch[38] - input_patch[41] + input_patch[42];
    trans_input_patch[23] = input_patch[21] - input_patch[23] + input_patch[25] - input_patch[27] + input_patch[37] - input_patch[39] + input_patch[41] - input_patch[43];
    trans_input_patch[24] = -input_patch[20] + input_patch[22] + input_patch[24] - input_patch[26] - input_patch[36] + input_patch[38] + input_patch[40] - input_patch[42];
    trans_input_patch[25] = -input_patch[21] - input_patch[22] + input_patch[25] + input_patch[26] - input_patch[37] - input_patch[38] + input_patch[41] + input_patch[42];
    trans_input_patch[26] = input_patch[21] - input_patch[22] - input_patch[25] + input_patch[26] + input_patch[37] - input_patch[38] - input_patch[41] + input_patch[42];
    trans_input_patch[27] = -input_patch[21] + input_patch[23] + input_patch[25] - input_patch[27] - input_patch[37] + input_patch[39] + input_patch[41] - input_patch[43];
    trans_input_patch[28] = input_patch[20] - input_patch[22] - input_patch[28] + input_patch[30] + input_patch[36] - input_patch[38] - input_patch[44] + input_patch[46];
    trans_input_patch[29] = input_patch[21] + input_patch[22] - input_patch[29] - input_patch[30] + input_patch[37] + input_patch[38] - input_patch[45] - input_patch[46];
    trans_input_patch[30] = -input_patch[21] + input_patch[22] + input_patch[29] - input_patch[30] - input_patch[37] + input_patch[38] + input_patch[45] - input_patch[46];
    trans_input_patch[31] = input_patch[21] - input_patch[23] - input_patch[29] + input_patch[31] + input_patch[37] - input_patch[39] - input_patch[45] + input_patch[47];
    trans_input_patch[32] = -input_patch[16] + input_patch[18] + input_patch[24] - input_patch[26] + input_patch[32] - input_patch[34] - input_patch[40] + input_patch[42];
    trans_input_patch[33] = -input_patch[17] - input_patch[18] + input_patch[25] + input_patch[26] + input_patch[33] + input_patch[34] - input_patch[41] - input_patch[42];
    trans_input_patch[34] = input_patch[17] - input_patch[18] - input_patch[25] + input_patch[26] - input_patch[33] + input_patch[34] + input_patch[41] - input_patch[42];
    trans_input_patch[35] = -input_patch[17] + input_patch[19] + input_patch[25] - input_patch[27] + input_patch[33] - input_patch[35] - input_patch[41] + input_patch[43];
    trans_input_patch[36] = -input_patch[20] + input_patch[22] - input_patch[24] + input_patch[26] + input_patch[36] - input_patch[38] + input_patch[40] - input_patch[42];
    trans_input_patch[37] = -input_patch[21] - input_patch[22] - input_patch[25] - input_patch[26] + input_patch[37] + input_patch[38] + input_patch[41] + input_patch[42];
    trans_input_patch[38] = input_patch[21] - input_patch[22] + input_patch[25] - input_patch[26] - input_patch[37] + input_patch[38] - input_patch[41] + input_patch[42];
    trans_input_patch[39] = -input_patch[21] + input_patch[23] - input_patch[25] + input_patch[27] + input_patch[37] - input_patch[39] + input_patch[41] - input_patch[43];
    trans_input_patch[40] = input_patch[20] - input_patch[22] - input_patch[24] + input_patch[26] - input_patch[36] + input_patch[38] + input_patch[40] - input_patch[42];
    trans_input_patch[41] = input_patch[21] + input_patch[22] - input_patch[25] - input_patch[26] - input_patch[37] - input_patch[38] + input_patch[41] + input_patch[42];
    trans_input_patch[42] = -input_patch[21] + input_patch[22] + input_patch[25] - input_patch[26] + input_patch[37] - input_patch[38] - input_patch[41] + input_patch[42];
    trans_input_patch[43] = input_patch[21] - input_patch[23] - input_patch[25] + input_patch[27] - input_patch[37] + input_patch[39] + input_patch[41] - input_patch[43];
    trans_input_patch[44] = -input_patch[20] + input_patch[22] + input_patch[28] - input_patch[30] + input_patch[36] - input_patch[38] - input_patch[44] + input_patch[46];
    trans_input_patch[45] = -input_patch[21] - input_patch[22] + input_patch[29] + input_patch[30] + input_patch[37] + input_patch[38] - input_patch[45] - input_patch[46];
    trans_input_patch[46] = input_patch[21] - input_patch[22] - input_patch[29] + input_patch[30] - input_patch[37] + input_patch[38] + input_patch[45] - input_patch[46];
    trans_input_patch[47] = -input_patch[21] + input_patch[23] + input_patch[29] - input_patch[31] + input_patch[37] - input_patch[39] - input_patch[45] + input_patch[47];
    trans_input_patch[48] = input_patch[16] - input_patch[18] - input_patch[24] + input_patch[26] - input_patch[48] + input_patch[50] + input_patch[56] - input_patch[58];
    trans_input_patch[49] = input_patch[17] + input_patch[18] - input_patch[25] - input_patch[26] - input_patch[49] - input_patch[50] + input_patch[57] + input_patch[58];
    trans_input_patch[50] = -input_patch[17] + input_patch[18] + input_patch[25] - input_patch[26] + input_patch[49] - input_patch[50] - input_patch[57] + input_patch[58];
    trans_input_patch[51] = input_patch[17] - input_patch[19] - input_patch[25] + input_patch[27] - input_patch[49] + input_patch[51] + input_patch[57] - input_patch[59];
    trans_input_patch[52] = input_patch[20] - input_patch[22] + input_patch[24] - input_patch[26] - input_patch[52] + input_patch[54] - input_patch[56] + input_patch[58];
    trans_input_patch[53] = input_patch[21] + input_patch[22] + input_patch[25] + input_patch[26] - input_patch[53] - input_patch[54] - input_patch[57] - input_patch[58];
    trans_input_patch[54] = -input_patch[21] + input_patch[22] - input_patch[25] + input_patch[26] + input_patch[53] - input_patch[54] + input_patch[57] - input_patch[58];
    trans_input_patch[55] = input_patch[21] - input_patch[23] + input_patch[25] - input_patch[27] - input_patch[53] + input_patch[55] - input_patch[57] + input_patch[59];
    trans_input_patch[56] = -input_patch[20] + input_patch[22] + input_patch[24] - input_patch[26] + input_patch[52] - input_patch[54] - input_patch[56] + input_patch[58];
    trans_input_patch[57] = -input_patch[21] - input_patch[22] + input_patch[25] + input_patch[26] + input_patch[53] + input_patch[54] - input_patch[57] - input_patch[58];
    trans_input_patch[58] = input_patch[21] - input_patch[22] - input_patch[25] + input_patch[26] - input_patch[53] + input_patch[54] + input_patch[57] - input_patch[58];
    trans_input_patch[59] = -input_patch[21] + input_patch[23] + input_patch[25] - input_patch[27] + input_patch[53] - input_patch[55] - input_patch[57] + input_patch[59];
    trans_input_patch[60] = input_patch[20] - input_patch[22] - input_patch[28] + input_patch[30] - input_patch[52] + input_patch[54] + input_patch[60] - input_patch[62];
    trans_input_patch[61] = input_patch[21] + input_patch[22] - input_patch[29] - input_patch[30] - input_patch[53] - input_patch[54] + input_patch[61] + input_patch[62];
    trans_input_patch[62] = -input_patch[21] + input_patch[22] + input_patch[29] - input_patch[30] + input_patch[53] - input_patch[54] - input_patch[61] + input_patch[62];
    trans_input_patch[63] = input_patch[21] - input_patch[23] - input_patch[29] + input_patch[31] - input_patch[53] + input_patch[55] + input_patch[61] - input_patch[63];


    int offset =  (((f_b * nD + by) * nH + ty) * nW + bx) * C + tx;
    int stride = B * nD * nH * nW * C;


    for(int i = 0; i < 64; i++) {
      wino_input [ i * stride + offset ] = T(trans_input_patch[i]);
    }
}

// dim3 threadsPerBlock(C, nH, 1)
// dim3 numBlocks(nW, nD, nB)
//wino_output = (64, Batch, nD, nH, nW, K)
//norm_output = (Batch, D, H, W, K)
template <typename T>
__global__ void outputWino2NormTransform3x3x3(const T *wino_output, T *norm_output, int B, int output_D, int output_H, int output_W, int K) {
    int bz = blockIdx.z; //b
    int by = blockIdx.y; //d
    int bx = blockIdx.x; //w
    int ty = threadIdx.y; //h
    int tx = threadIdx.x; //K
    int nH, nW, nD;
    nH = (output_H + 1) / 2;
    nW = (output_W + 1) / 2;
    nD = (output_D + 1) / 2;


    T product_patch[64] = {T(0)};

    for(int i = 0; i < 64; i++) {
      product_patch[i] = wino_output[ ((((i * B + bz) * nD + by) * nH + ty) * nW + bx) * K + tx];
}

    T output_patch[8] = {T(0)};
    output_patch[0] = product_patch[0] + product_patch[10] + product_patch[16] + product_patch[17] + product_patch[18] + product_patch[1] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[24] + product_patch[25] + product_patch[26] + product_patch[2] + product_patch[32] + product_patch[33] + product_patch[34] + product_patch[36] + product_patch[37] + product_patch[38] + product_patch[40] + product_patch[41] + product_patch[42] + product_patch[4] + product_patch[5] + product_patch[6] + product_patch[8] + product_patch[9];
    output_patch[1] = -product_patch[10] - product_patch[11] + product_patch[17] - product_patch[18] - product_patch[19] + product_patch[1] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[25] - product_patch[26] - product_patch[27] - product_patch[2] + product_patch[33] - product_patch[34] - product_patch[35] + product_patch[37] - product_patch[38] - product_patch[39] - product_patch[3] + product_patch[41] - product_patch[42] - product_patch[43] + product_patch[5] - product_patch[6] - product_patch[7] + product_patch[9];
    output_patch[2] = -product_patch[10] - product_patch[12] - product_patch[13] - product_patch[14] + product_patch[20] + product_patch[21] + product_patch[22] - product_patch[24] - product_patch[25] - product_patch[26] - product_patch[28] - product_patch[29] - product_patch[30] + product_patch[36] + product_patch[37] + product_patch[38] - product_patch[40] - product_patch[41] - product_patch[42] - product_patch[44] - product_patch[45] - product_patch[46] + product_patch[4] + product_patch[5] + product_patch[6] - product_patch[8] - product_patch[9];
    output_patch[3] = product_patch[10] + product_patch[11] - product_patch[13] + product_patch[14] + product_patch[15] + product_patch[21] - product_patch[22] - product_patch[23] - product_patch[25] + product_patch[26] + product_patch[27] - product_patch[29] + product_patch[30] + product_patch[31] + product_patch[37] - product_patch[38] - product_patch[39] - product_patch[41] + product_patch[42] + product_patch[43] - product_patch[45] + product_patch[46] + product_patch[47] + product_patch[5] - product_patch[6] - product_patch[7] - product_patch[9];
    output_patch[4] = product_patch[16] + product_patch[17] + product_patch[18] + product_patch[20] + product_patch[21] + product_patch[22] + product_patch[24] + product_patch[25] + product_patch[26] - product_patch[32] - product_patch[33] - product_patch[34] - product_patch[36] - product_patch[37] - product_patch[38] - product_patch[40] - product_patch[41] - product_patch[42] - product_patch[48] - product_patch[49] - product_patch[50] - product_patch[52] - product_patch[53] - product_patch[54] - product_patch[56] - product_patch[57] - product_patch[58];
    output_patch[5] = product_patch[17] - product_patch[18] - product_patch[19] + product_patch[21] - product_patch[22] - product_patch[23] + product_patch[25] - product_patch[26] - product_patch[27] - product_patch[33] + product_patch[34] + product_patch[35] - product_patch[37] + product_patch[38] + product_patch[39] - product_patch[41] + product_patch[42] + product_patch[43] - product_patch[49] + product_patch[50] + product_patch[51] - product_patch[53] + product_patch[54] + product_patch[55] - product_patch[57] + product_patch[58] + product_patch[59];
    output_patch[6] = product_patch[20] + product_patch[21] + product_patch[22] - product_patch[24] - product_patch[25] - product_patch[26] - product_patch[28] - product_patch[29] - product_patch[30] - product_patch[36] - product_patch[37] - product_patch[38] + product_patch[40] + product_patch[41] + product_patch[42] + product_patch[44] + product_patch[45] + product_patch[46] - product_patch[52] - product_patch[53] - product_patch[54] + product_patch[56] + product_patch[57] + product_patch[58] + product_patch[60] + product_patch[61] + product_patch[62];
    output_patch[7] = product_patch[21] - product_patch[22] - product_patch[23] - product_patch[25] + product_patch[26] + product_patch[27] - product_patch[29] + product_patch[30] + product_patch[31] - product_patch[37] + product_patch[38] + product_patch[39] + product_patch[41] - product_patch[42] - product_patch[43] + product_patch[45] - product_patch[46] - product_patch[47] - product_patch[53] + product_patch[54] + product_patch[55] + product_patch[57] - product_patch[58] - product_patch[59] + product_patch[61] - product_patch[62] - product_patch[63];


    norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[0];
    if(output_W % 2 == 0 || bx != nW - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[1];
    if(output_H % 2 == 0 || ty != nH - 1)
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[2];
    if((output_W % 2 == 0 || bx != nW - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 0)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[3];
    if(output_D % 2 == 0 || by != nD - 1)
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[4];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 0)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[5];
    if((output_D % 2 == 0 || by != nD - 1) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 0)) * K + tx] = output_patch[6];
    if((output_D % 2 == 0 || by != nD - 1) && (output_W % 2 == 0 || bx != nW - 1 ) && (output_H % 2 == 0 || ty != nH - 1))
      norm_output[((( bz * output_D + (2 * by + 1)) * output_H + (2 * ty + 1)) * output_W + (2 * bx + 1)) * K + tx] = output_patch[7];
}

