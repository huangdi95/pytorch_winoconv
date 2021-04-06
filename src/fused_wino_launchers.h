#ifndef FUSED_WINO_LAUNCHERS_H_
#define FUSED_WINO_LAUNCHERS_H_
#include <iostream>
#include "src/base_conv_launchers.h"

template <typename T>
void fusedLauncherStrideOne2D(
    const T *input,
    const T *weight,
    T *output,
    T *tmp_weight_buffer,
    int B, int H, int W, int C, int K, int kernel_H, int kernel_W, int pad_h, int pad_w,
    int num_split,
    int *H_start_gpu,
    int *H_end_gpu,
    int *W_start_gpu,
    int *W_end_gpu,
    int *kernel_stride_gpu,
    float *tmp_out_buffer);

static void splitControlFused2D(int H, int W, int *num_split, int **H_start, int **H_end, int **W_start, int **W_end) {
    if (H == 1 && W == 1) {
        int H_s[] = {0}; 
        int H_e[] = {1}; 
        int W_s[] = {0}; 
        int W_e[] = {1}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 1 && W == 3) {
        int H_s[] = {0}; 
        int H_e[] = {1}; 
        int W_s[] = {0}; 
        int W_e[] = {3}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 1 && W == 4) {
        int H_s[] = {0, 0}; 
        int H_e[] = {1, 1}; 
        int W_s[] = {0, 2}; 
        int W_e[] = {2, 4}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 1 && W == 5) {
        int H_s[] = {0, 0}; 
        int H_e[] = {1, 1}; 
        int W_s[] = {0, 3}; 
        int W_e[] = {3, 5}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 1 && W == 7) {
        int H_s[] = {0, 0, 0}; 
        int H_e[] = {1, 1, 1}; 
        int W_s[] = {0, 3, 6}; 
        int W_e[] = {3, 6, 7}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 3 && W == 1) {
        int H_s[] = {0}; 
        int H_e[] = {3}; 
        int W_s[] = {0}; 
        int W_e[] = {1}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 3 && W == 3) {
        int H_s[] = {0}; 
        int H_e[] = {3}; 
        int W_s[] = {0}; 
        int W_e[] = {3}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 3 && W == 4) {
        int H_s[] = {0, 0}; 
        int H_e[] = {3, 3}; 
        int W_s[] = {0, 2}; 
        int W_e[] = {2, 4}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 3 && W == 5) {
        int H_s[] = {0, 0}; 
        int H_e[] = {3, 3}; 
        int W_s[] = {0, 3}; 
        int W_e[] = {3, 5}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 4 && W == 3) {
        int H_s[] = {0, 2}; 
        int H_e[] = {2, 4}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {3, 3}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 4 && W == 4) {
        int H_s[] = {0, 0, 2, 2}; 
        int H_e[] = {2, 2, 4, 4}; 
        int W_s[] = {0, 2, 0, 2}; 
        int W_e[] = {2, 4, 2, 4}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 5 && W == 5) {
        int H_s[] = {0, 0, 3, 3}; 
        int H_e[] = {3, 3, 5, 5}; 
        int W_s[] = {0, 3, 0, 3}; 
        int W_e[] = {3, 5, 3, 5}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 6 && W == 6) {
        int H_s[] = {0, 0, 3, 3};
        int H_e[] = {3, 3, 6, 6};
        int W_s[] = {0, 3, 0, 3};
        int W_e[] = {3, 6, 3, 6};
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 7 && W == 7) {
        int H_s[] = {0, 0, 3, 3, 0, 3, 6, 6, 6};
        int H_e[] = {3, 3, 6, 6, 3, 6, 7, 7, 7};
        int W_s[] = {0, 3, 0, 3, 6, 6, 0, 3, 6};
        int W_e[] = {3, 6, 3, 6, 7, 7, 3, 6, 7};
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else if (H == 9 && W == 9) {
        int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6}; 
        int H_e[] = {3, 3, 3, 6, 6, 6, 9, 9, 9}; 
        int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6}; 
        int W_e[] = {3, 6, 9, 3, 6, 9, 3, 6, 9}; 
        *num_split = sizeof(H_s) / sizeof(H_s[0]); 
        tensorCopy2D(H_s, W_s, H_e, W_e, num_split, H_start, H_end, W_start, W_end);
    } else {
      std::cout << H << "x" << W << " not implemented yet." << std::endl;
      exit(-1);
    }
}
#endif //FUSED_WINO_LAUNCHERS_H_
