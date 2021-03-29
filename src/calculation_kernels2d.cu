template <typename T>
__device__ __forceinline__ void wNorm2WinoCalculation2D(T *input_patch, T* output_patch, int H, int W) {
    if(H == 1 && W == 1) {
        output_patch[0] = input_patch[0];
        output_patch[1] = input_patch[0];
        output_patch[2] = input_patch[0];
        output_patch[3] = input_patch[0];
    } else if(H == 1 && W == 2) {
        output_patch[0] = input_patch[0];
        output_patch[1] = input_patch[0] + input_patch[1];
        output_patch[2] = input_patch[1];
        output_patch[3] = input_patch[0];
        output_patch[4] = input_patch[0] + input_patch[1];
        output_patch[5] = input_patch[1];
    } else if(H == 1 && W == 3) {
        output_patch[0] = input_patch[0];
        output_patch[1] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2]);
        output_patch[2] = T(1/2.)*(input_patch[0] - input_patch[1] + input_patch[2]);
        output_patch[3] = input_patch[2];
        output_patch[4] = input_patch[0];
        output_patch[5] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2]);
        output_patch[6] = T(1/2.)*(input_patch[0] - input_patch[1] + input_patch[2]);
        output_patch[7] = input_patch[2];
    } else if(H == 2 && W == 1) {
        output_patch[0] = input_patch[0];
        output_patch[1] = input_patch[0];
        output_patch[2] = input_patch[0] + input_patch[1];
        output_patch[3] = input_patch[0] + input_patch[1];
        output_patch[4] = input_patch[1];
        output_patch[5] = input_patch[1];
    } else if(H == 2 && W == 2) {
        output_patch[0] = input_patch[0];
        output_patch[1] = input_patch[0] + input_patch[1];
        output_patch[2] = input_patch[1];
        output_patch[3] = input_patch[0] + input_patch[2];
        output_patch[4] = input_patch[0] + input_patch[1] + input_patch[2] + input_patch[3];
        output_patch[5] = input_patch[1] + input_patch[3];
        output_patch[6] = input_patch[2];
        output_patch[7] = input_patch[2] + input_patch[3];
        output_patch[8] = input_patch[3];
    } else if(H == 2 && W == 3) {
        output_patch[0] = input_patch[0];
        output_patch[1] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2]);
        output_patch[2] = T(1/2.)*(input_patch[0] - input_patch[1] + input_patch[2]);
        output_patch[3] = input_patch[2];
        output_patch[4] = input_patch[0] + input_patch[3];
        output_patch[5] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2] + input_patch[3] + input_patch[4] + input_patch[5]);
        output_patch[6] = T(1/2.)*(input_patch[0] - input_patch[1] + input_patch[2] + input_patch[3] - input_patch[4] + input_patch[5]);
        output_patch[7] = input_patch[2] + input_patch[5];
        output_patch[8] = input_patch[3];
        output_patch[9] = T(1/2.)*(input_patch[3] + input_patch[4] + input_patch[5]);
        output_patch[10] = T(1/2.)*(input_patch[3] - input_patch[4] + input_patch[5]);
        output_patch[11] = input_patch[5];
    } else if(H == 3 && W == 1) {
        output_patch[0] = input_patch[0];
        output_patch[1] = input_patch[0];
        output_patch[2] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2]);
        output_patch[3] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2]);
        output_patch[4] = T(1/2.)*(input_patch[0] - input_patch[1] + input_patch[2]);
        output_patch[5] = T(1/2.)*(input_patch[0] - input_patch[1] + input_patch[2]);
        output_patch[6] = input_patch[2];
        output_patch[7] = input_patch[2];
    } else if(H == 3 && W == 2) {
        output_patch[0] = input_patch[0];
        output_patch[1] = input_patch[0] + input_patch[1];
        output_patch[2] = input_patch[1];
        output_patch[3] = T(1/2.)*(input_patch[0] + input_patch[2] + input_patch[4]);
        output_patch[4] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2] + input_patch[3] + input_patch[4] + input_patch[5]);
        output_patch[5] = T(1/2.)*(input_patch[1] + input_patch[3] + input_patch[5]);
        output_patch[6] = T(1/2.)*(input_patch[0] - input_patch[2] + input_patch[4]);
        output_patch[7] = T(1/2.)*(input_patch[0] + input_patch[1] - input_patch[2] - input_patch[3] + input_patch[4] + input_patch[5]);
        output_patch[8] = T(1/2.)*(input_patch[1] - input_patch[3] + input_patch[5]);
        output_patch[9] = input_patch[4];
        output_patch[10] = input_patch[4] + input_patch[5];
        output_patch[11] = input_patch[5];
    } else if(H == 3 && W == 3) {
        output_patch[0] = input_patch[0];
        output_patch[1] = T(1/2.)*(input_patch[0] + input_patch[1] + input_patch[2]);
        output_patch[2] = T(1/2.)*(input_patch[0] - input_patch[1] + input_patch[2]);
        output_patch[3] = input_patch[2];
        output_patch[4] = T(1/2.)*(input_patch[0] + input_patch[3] + input_patch[6]);
        output_patch[5] = T(1/4.)*(input_patch[0] + input_patch[1] + input_patch[2] + input_patch[3] + input_patch[4] + input_patch[5] + input_patch[6] + input_patch[7] + input_patch[8]);
        output_patch[6] = T(1/4.)*(input_patch[0] - input_patch[1] + input_patch[2] + input_patch[3] - input_patch[4] + input_patch[5] + input_patch[6] - input_patch[7] + input_patch[8]);
        output_patch[7] = T(1/2.)*(input_patch[2] + input_patch[5] + input_patch[8]);
        output_patch[8] = T(1/2.)*(input_patch[0] - input_patch[3] + input_patch[6]);
        output_patch[9] = T(1/4.)*(input_patch[0] + input_patch[1] + input_patch[2] - input_patch[3] - input_patch[4] - input_patch[5] + input_patch[6] + input_patch[7] + input_patch[8]);
        output_patch[10] = T(1/4.)*(input_patch[0] - input_patch[1] + input_patch[2] - input_patch[3] + input_patch[4] - input_patch[5] + input_patch[6] - input_patch[7] + input_patch[8]);
        output_patch[11] = T(1/2.)*(input_patch[2] - input_patch[5] + input_patch[8]);
        output_patch[12] = input_patch[6];
        output_patch[13] = T(1/2.)*(input_patch[6] + input_patch[7] + input_patch[8]);
        output_patch[14] = T(1/2.)*(input_patch[6] - input_patch[7] + input_patch[8]);
        output_patch[15] = input_patch[8];
    }
}

template <typename T>
__device__ void inputNorm2WinoCalculation2D(T *input_patch, T *output_patch, int H, int W) {
//        output_patch[0] = input_patch[0]; 
//        output_patch[1] = input_patch[1]; 
//        output_patch[2] = input_patch[2]; 
//        output_patch[3] = input_patch[3]; 
//        output_patch[4] = input_patch[4]; 
//        output_patch[5] = input_patch[5]; 
//        output_patch[6] = input_patch[6]; 
//        output_patch[7] = input_patch[7]; 
//        output_patch[8] = input_patch[8]; 
//        output_patch[9] = input_patch[9]; 
//        output_patch[10] = input_patch[10];
//        output_patch[11] = input_patch[11];
//        output_patch[12] = input_patch[12];
//        output_patch[13] = input_patch[13];
//        output_patch[14] = input_patch[14];
//        output_patch[15] = input_patch[15];
    if(H == 1 && W == 1) {
        output_patch[0] = input_patch[0];
        output_patch[1] = input_patch[1];
        output_patch[2] = input_patch[2];
        output_patch[3] = input_patch[3];
    } else if(H == 1 && W == 2) {
        output_patch[0] = input_patch[0] - input_patch[1];
        output_patch[1] = input_patch[1];
        output_patch[2] = -input_patch[1] + input_patch[2];
        output_patch[3] = input_patch[3] - input_patch[4];
        output_patch[4] = input_patch[4];
        output_patch[5] = -input_patch[4] + input_patch[5];
    } else if(H == 1 && W == 3) {
        output_patch[0] = input_patch[0] - input_patch[2];
        output_patch[1] = input_patch[1] + input_patch[2];
        output_patch[2] = -input_patch[1] + input_patch[2];
        output_patch[3] = input_patch[1] - input_patch[3];
        output_patch[4] = input_patch[4] - input_patch[6];
        output_patch[5] = input_patch[5] + input_patch[6];
        output_patch[6] = -input_patch[5] + input_patch[6];
        output_patch[7] = input_patch[5] - input_patch[7];
    } else if(H == 2 && W == 1) {
        output_patch[0] = input_patch[0] - input_patch[2];
        output_patch[1] = input_patch[1] - input_patch[3];
        output_patch[2] = input_patch[2];
        output_patch[3] = input_patch[3];
        output_patch[4] = -input_patch[2] + input_patch[4];
        output_patch[5] = -input_patch[3] + input_patch[5];
    } else if(H == 2 && W == 2) {
        output_patch[0] = input_patch[0] - input_patch[1] - input_patch[3] + input_patch[4];
        output_patch[1] = input_patch[1] - input_patch[4];
        output_patch[2] = -input_patch[1] + input_patch[2] + input_patch[4] - input_patch[5];
        output_patch[3] = input_patch[3] - input_patch[4];
        output_patch[4] = input_patch[4];
        output_patch[5] = -input_patch[4] + input_patch[5];
        output_patch[6] = -input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
        output_patch[7] = -input_patch[4] + input_patch[7];
        output_patch[8] = input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
    } else if(H == 2 && W == 3) {
        output_patch[0] = input_patch[0] - input_patch[2] - input_patch[4] + input_patch[6];
        output_patch[1] = input_patch[1] + input_patch[2] - input_patch[5] - input_patch[6];
        output_patch[2] = -input_patch[1] + input_patch[2] + input_patch[5] - input_patch[6];
        output_patch[3] = input_patch[1] - input_patch[3] - input_patch[5] + input_patch[7];
        output_patch[4] = input_patch[4] - input_patch[6];
        output_patch[5] = input_patch[5] + input_patch[6];
        output_patch[6] = -input_patch[5] + input_patch[6];
        output_patch[7] = input_patch[5] - input_patch[7];
        output_patch[8] = -input_patch[10] - input_patch[4] + input_patch[6] + input_patch[8];
        output_patch[9] = input_patch[10] - input_patch[5] - input_patch[6] + input_patch[9];
        output_patch[10] = input_patch[10] + input_patch[5] - input_patch[6] - input_patch[9];
        output_patch[11] = -input_patch[11] - input_patch[5] + input_patch[7] + input_patch[9];
    } else if(H == 3 && W == 1) {
        output_patch[0] = input_patch[0] - input_patch[4];
        output_patch[1] = input_patch[1] - input_patch[5];
        output_patch[2] = input_patch[2] + input_patch[4];
        output_patch[3] = input_patch[3] + input_patch[5];
        output_patch[4] = -input_patch[2] + input_patch[4];
        output_patch[5] = -input_patch[3] + input_patch[5];
        output_patch[6] = input_patch[2] - input_patch[6];
        output_patch[7] = input_patch[3] - input_patch[7];
    } else if(H == 3 && W == 2) {
        output_patch[0] = input_patch[0] - input_patch[1] - input_patch[6] + input_patch[7];
        output_patch[1] = input_patch[1] - input_patch[7];
        output_patch[2] = -input_patch[1] + input_patch[2] + input_patch[7] - input_patch[8];
        output_patch[3] = input_patch[3] - input_patch[4] + input_patch[6] - input_patch[7];
        output_patch[4] = input_patch[4] + input_patch[7];
        output_patch[5] = -input_patch[4] + input_patch[5] - input_patch[7] + input_patch[8];
        output_patch[6] = -input_patch[3] + input_patch[4] + input_patch[6] - input_patch[7];
        output_patch[7] = -input_patch[4] + input_patch[7];
        output_patch[8] = input_patch[4] - input_patch[5] - input_patch[7] + input_patch[8];
        output_patch[9] = input_patch[10] + input_patch[3] - input_patch[4] - input_patch[9];
        output_patch[10] = -input_patch[10] + input_patch[4];
        output_patch[11] = input_patch[10] - input_patch[11] - input_patch[4] + input_patch[5];
    } else if(H == 3 && W == 3) {
        output_patch[0] = input_patch[0] + input_patch[10] - input_patch[2] - input_patch[8];
        output_patch[1] = -input_patch[10] + input_patch[1] + input_patch[2] - input_patch[9];
        output_patch[2] = -input_patch[10] - input_patch[1] + input_patch[2] + input_patch[9];
        output_patch[3] = input_patch[11] + input_patch[1] - input_patch[3] - input_patch[9];
        output_patch[4] = -input_patch[10] + input_patch[4] - input_patch[6] + input_patch[8];
        output_patch[5] = input_patch[10] + input_patch[5] + input_patch[6] + input_patch[9];
        output_patch[6] = input_patch[10] - input_patch[5] + input_patch[6] - input_patch[9];
        output_patch[7] = -input_patch[11] + input_patch[5] - input_patch[7] + input_patch[9];
        output_patch[8] = -input_patch[10] - input_patch[4] + input_patch[6] + input_patch[8];
        output_patch[9] = input_patch[10] - input_patch[5] - input_patch[6] + input_patch[9];
        output_patch[10] = input_patch[10] + input_patch[5] - input_patch[6] - input_patch[9];
        output_patch[11] = -input_patch[11] - input_patch[5] + input_patch[7] + input_patch[9];
        output_patch[12] = -input_patch[12] + input_patch[14] + input_patch[4] - input_patch[6];
        output_patch[13] = -input_patch[13] - input_patch[14] + input_patch[5] + input_patch[6];
        output_patch[14] = input_patch[13] - input_patch[14] - input_patch[5] + input_patch[6];
        output_patch[15] = -input_patch[13] + input_patch[15] + input_patch[5] - input_patch[7];
    }
}

template <typename T>
__device__ void outputWino2NormCalculation2D(const T *input_patch, T *output_patch, int H, int W) {
//    output_patch[0] = input_patch[0];
//    output_patch[1] = input_patch[1];
//    output_patch[2] = input_patch[2];
//    output_patch[3] = input_patch[3];
    if(H == 1 && W == 1) {
    output_patch[0] = input_patch[0];
    output_patch[1] = input_patch[1];
    output_patch[2] = input_patch[2];
    output_patch[3] = input_patch[3];
    } else if(H == 1 && W == 2) {
    output_patch[0] = input_patch[0] + input_patch[1];
    output_patch[1] = input_patch[1] + input_patch[2];
    output_patch[2] = input_patch[3] + input_patch[4];
    output_patch[3] = input_patch[4] + input_patch[5];
    } else if(H == 1 && W == 3) {
    output_patch[0] = input_patch[0] + input_patch[1] + input_patch[2];
    output_patch[1] = input_patch[1] - input_patch[2] - input_patch[3];
    output_patch[2] = input_patch[4] + input_patch[5] + input_patch[6];
    output_patch[3] = input_patch[5] - input_patch[6] - input_patch[7];
    } else if(H == 2 && W == 1) {
    output_patch[0] = input_patch[0] + input_patch[2];
    output_patch[1] = input_patch[1] + input_patch[3];
    output_patch[2] = input_patch[2] + input_patch[4];
    output_patch[3] = input_patch[3] + input_patch[5];
    } else if(H == 2 && W == 2) {
    output_patch[0] = input_patch[0] + input_patch[1] + input_patch[3] + input_patch[4];
    output_patch[1] = input_patch[1] + input_patch[2] + input_patch[4] + input_patch[5];
    output_patch[2] = input_patch[3] + input_patch[4] + input_patch[6] + input_patch[7];
    output_patch[3] = input_patch[4] + input_patch[5] + input_patch[7] + input_patch[8];
    } else if(H == 2 && W == 3) {
    output_patch[0] = input_patch[0] + input_patch[1] + input_patch[2] + input_patch[4] + input_patch[5] + input_patch[6];
    output_patch[1] = input_patch[1] - input_patch[2] - input_patch[3] + input_patch[5] - input_patch[6] - input_patch[7];
    output_patch[2] = input_patch[10] + input_patch[4] + input_patch[5] + input_patch[6] + input_patch[8] + input_patch[9];
    output_patch[3] = -input_patch[10] - input_patch[11] + input_patch[5] - input_patch[6] - input_patch[7] + input_patch[9];
    } else if(H == 3 && W == 1) {
    output_patch[0] = input_patch[0] + input_patch[2] + input_patch[4];
    output_patch[1] = input_patch[1] + input_patch[3] + input_patch[5];
    output_patch[2] = input_patch[2] - input_patch[4] - input_patch[6];
    output_patch[3] = input_patch[3] - input_patch[5] - input_patch[7];
    } else if(H == 3 && W == 2) {
    output_patch[0] = input_patch[0] + input_patch[1] + input_patch[3] + input_patch[4] + input_patch[6] + input_patch[7];
    output_patch[1] = input_patch[1] + input_patch[2] + input_patch[4] + input_patch[5] + input_patch[7] + input_patch[8];
    output_patch[2] = -input_patch[10] + input_patch[3] + input_patch[4] - input_patch[6] - input_patch[7] - input_patch[9];
    output_patch[3] = -input_patch[10] - input_patch[11] + input_patch[4] + input_patch[5] - input_patch[7] - input_patch[8];
    } else if(H == 3 && W == 3) {
//    output_patch[0] = input_patch[5];
//    output_patch[1] = input_patch[5];
//    output_patch[2] = input_patch[5];
//    output_patch[3] = input_patch[5];
    output_patch[0] = input_patch[0] + input_patch[10] + input_patch[1] + input_patch[2] + input_patch[4] + input_patch[5] + input_patch[6] + input_patch[8] + input_patch[9];
    output_patch[1] = -input_patch[10] - input_patch[11] + input_patch[1] - input_patch[2] - input_patch[3] + input_patch[5] - input_patch[6] - input_patch[7] + input_patch[9];
    output_patch[2] = -input_patch[10] - input_patch[12] - input_patch[13] - input_patch[14] + input_patch[4] + input_patch[5] + input_patch[6] - input_patch[8] - input_patch[9];
    output_patch[3] = input_patch[10] + input_patch[11] - input_patch[13] + input_patch[14] + input_patch[15] + input_patch[5] - input_patch[6] - input_patch[7] - input_patch[9];
    }
//     else {
//    output_patch[0] = input_patch[0] + input_patch[10] + input_patch[1] + input_patch[2] + input_patch[4] + input_patch[5] + input_patch[6] + input_patch[8] + input_patch[9];
//    output_patch[1] = -input_patch[10] - input_patch[11] + input_patch[1] - input_patch[2] - input_patch[3] + input_patch[5] - input_patch[6] - input_patch[7] + input_patch[9];
//    output_patch[2] = -input_patch[10] - input_patch[12] - input_patch[13] - input_patch[14] + input_patch[4] + input_patch[5] + input_patch[6] - input_patch[8] - input_patch[9];
//    output_patch[3] = input_patch[10] + input_patch[11] - input_patch[13] + input_patch[14] + input_patch[15] + input_patch[5] - input_patch[6] - input_patch[7] - input_patch[9];
//    }

}

