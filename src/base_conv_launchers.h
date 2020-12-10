#ifndef BASE_CONV_LAUNCHERS_H_
#define BASE_CONV_LAUNCHERS_H_
#include <iostream>

template <typename T>
void convLauncherStrideOne3x2(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
template <typename T>
void convLauncherStrideOne3x3(const T *input, const T *weight,
                              T *tmp_input_buffer, T *tmp_weight_buffer,
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int B, int D, int H, int W, int C, int K,
                              int kernel_D, int kernel_H, int kernel_W, int pad_d, int pad_h, int pad_w,
                              T *output);

template <typename T>
void transform(const T *input, const T *weight, const T *tmp_product_buffer,
                   T *tmp_input_buffer, T *tmp_weight_buffer, T *output,
                   int B, int D, int H, int W, int C, int K,
                   int kernel_D, int kernel_H, int kernel_W, int pad_d, int pad_h, int pad_w);
template <typename T>
void split(const T *input, const T *weight, const T *tmp_product_buffer,
                   T *tmp_input_buffer, T *tmp_weight_buffer, T *output,
                   int B, int D, int H, int W, int C, int K,
                   int kernel_D, int kernel_H, int kernel_W, int pad_d, int pad_h, int pad_w);
template <typename T>
void convLauncherStrideOneLarge(const T *input, const T *weight,
                              T *tmp_input_buffer, T *tmp_weight_buffer,
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int B, int D, int H, int W, int C, int K,
                              int kernel_D, int kernel_H, int kernel_W, int pad_d, int pad_h, int pad_w,
                              T *output);
//template <typename T>
static void tensorCopy(int *D_s, int *H_s, int *W_s, int *D_e, int *H_e, int *W_e, int *num_split, int **D_start, int **D_end, int **H_start, int **H_end, int **W_start, int **W_end) {
    *D_start = new int[*num_split];
    *H_start = new int[*num_split];
    *W_start = new int[*num_split];
    *D_end = new int[*num_split];
    *H_end = new int[*num_split];
    *W_end = new int[*num_split];
    std::copy(D_s, D_s + *num_split, *D_start);
    std::copy(H_s, H_s + *num_split, *H_start);
    std::copy(W_s, W_s + *num_split, *W_start);
    std::copy(D_e, D_e + *num_split, *D_end);
    std::copy(H_e, H_e + *num_split, *H_end);
    std::copy(W_e, W_e + *num_split, *W_end);
}

//template <typename T>
static void splitControl(int D, int H, int W, int *num_split, int **D_start, int **D_end, int **H_start, int **H_end, int **W_start, int **W_end) {
    if (D == 1 && H == 1 && W == 1) {
        int D_s[] = {0}; 
        int D_e[] = {1}; 
        int H_s[] = {0}; 
        int H_e[] = {1}; 
        int W_s[] = {0}; 
        int W_e[] = {1}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 1 && H == 3 && W == 4) {
        int D_s[] = {0, 0}; 
        int D_e[] = {1, 1}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {3, 3}; 
        int W_s[] = {0, 2}; 
        int W_e[] = {2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 1 && H == 4 && W == 3) {
        int D_s[] = {0, 0}; 
        int D_e[] = {1, 1}; 
        int H_s[] = {0, 2}; 
        int H_e[] = {2, 4}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {3, 3}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 1 && H == 4 && W == 4) {
        int D_s[] = {0, 0, 0, 0}; 
        int D_e[] = {1, 1, 1, 1}; 
        int H_s[] = {0, 0, 2, 2}; 
        int H_e[] = {2, 2, 4, 4}; 
        int W_s[] = {0, 2, 0, 2}; 
        int W_e[] = {2, 4, 2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 1 && H == 5 && W == 5) {
        int D_s[] = {0, 0, 0, 0}; 
        int D_e[] = {1, 1, 1, 1}; 
        int H_s[] = {0, 0, 3, 3}; 
        int H_e[] = {3, 3, 5, 5}; 
        int W_s[] = {0, 3, 0, 3}; 
        int W_e[] = {3, 5, 3, 5}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 1 && H == 7 && W == 7) {
        int D_s[] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; 
        int D_e[] = {1, 1, 1, 1, 1, 1, 1, 1, 1}; 
        int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6}; 
        int H_e[] = {3, 3, 3, 6, 6, 6, 7, 7, 7}; 
        int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6}; 
        int W_e[] = {3, 6, 7, 3, 6, 7, 3, 6, 7}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 1 && H == 9 && W == 9) {
        int D_s[] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; 
        int D_e[] = {1, 1, 1, 1, 1, 1, 1, 1, 1}; 
        int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6}; 
        int H_e[] = {3, 3, 3, 6, 6, 6, 9, 9, 9}; 
        int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6}; 
        int W_e[] = {3, 6, 9, 3, 6, 9, 3, 6, 9}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 2 && H == 2 && W == 2) {
        int D_s[] = {0};
        int D_e[] = {2};
        int H_s[] = {0};
        int H_e[] = {2};
        int W_s[] = {0};
        int W_e[] = {2};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 2 && H == 2 && W == 3) {
        int D_s[] = {0};
        int D_e[] = {2};
        int H_s[] = {0};
        int H_e[] = {2};
        int W_s[] = {0};
        int W_e[] = {3};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 2 && H == 3 && W == 2) {
        int D_s[] = {0};
        int D_e[] = {2};
        int H_s[] = {0};
        int H_e[] = {3};
        int W_s[] = {0};
        int W_e[] = {2};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 2 && H == 3 && W == 3) {
        int D_s[] = {0};
        int D_e[] = {2};
        int H_s[] = {0};
        int H_e[] = {3};
        int W_s[] = {0};
        int W_e[] = {3};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 2 && H == 3 && W == 4) {
        int D_s[] = {0, 0}; 
        int D_e[] = {2, 2}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {3, 3}; 
        int W_s[] = {0, 2}; 
        int W_e[] = {2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 2 && H == 4 && W == 3) {
        int D_s[] = {0, 0}; 
        int D_e[] = {2, 2}; 
        int H_s[] = {0, 2}; 
        int H_e[] = {2, 4}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {3, 3}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 2 && H == 4 && W == 4) {
        int D_s[] = {0, 0, 0, 0}; 
        int D_e[] = {2, 2, 2, 2}; 
        int H_s[] = {0, 0, 2, 2}; 
        int H_e[] = {2, 2, 4, 4}; 
        int W_s[] = {0, 2, 0, 2}; 
        int W_e[] = {2, 4, 2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 2 && W == 2) {
        int D_s[] = {0};
        int D_e[] = {3};
        int H_s[] = {0};
        int H_e[] = {2};
        int W_s[] = {0};
        int W_e[] = {2};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 2 && W == 3) {
        int D_s[] = {0};
        int D_e[] = {3};
        int H_s[] = {0};
        int H_e[] = {2};
        int W_s[] = {0};
        int W_e[] = {3};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 3 && W == 2) {
        int D_s[] = {0};
        int D_e[] = {3};
        int H_s[] = {0};
        int H_e[] = {3};
        int W_s[] = {0};
        int W_e[] = {2};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 3 && W == 3) {
        int D_s[] = {0};
        int D_e[] = {3};
        int H_s[] = {0};
        int H_e[] = {3};
        int W_s[] = {0};
        int W_e[] = {3};
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 3 && W == 4) {
        int D_s[] = {0, 0}; 
        int D_e[] = {3, 3}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {3, 3}; 
        int W_s[] = {0, 2}; 
        int W_e[] = {2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 4 && W == 3) {
        int D_s[] = {0, 0}; 
        int D_e[] = {3, 3}; 
        int H_s[] = {0, 2}; 
        int H_e[] = {2, 4}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {3, 3}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 4 && W == 4) {
        int D_s[] = {0, 0, 0, 0}; 
        int D_e[] = {3, 3, 3, 3}; 
        int H_s[] = {0, 0, 2, 2}; 
        int H_e[] = {2, 2, 4, 4}; 
        int W_s[] = {0, 2, 0, 2}; 
        int W_e[] = {2, 4, 2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 5 && W == 5) {
        int D_s[] = {0, 0, 0, 0}; 
        int D_e[] = {3, 3, 3, 3}; 
        int H_s[] = {0, 0, 3, 3}; 
        int H_e[] = {3, 3, 5, 5}; 
        int W_s[] = {0, 3, 0, 3}; 
        int W_e[] = {3, 5, 3, 5}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 3 && H == 7 && W == 7) {
        int D_s[] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; 
        int D_e[] = {3, 3, 3, 3, 3, 3, 3, 3, 3}; 
        int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6}; 
        int H_e[] = {3, 3, 3, 6, 6, 6, 7, 7, 7}; 
        int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6}; 
        int W_e[] = {3, 6, 7, 3, 6, 7, 3, 6, 7}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
    }else if(D == 4 && H == 2 && W == 2) {
        int D_s[] = {0, 2}; 
        int D_e[] = {2, 4}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {2, 2}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {2, 2}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    }else if(D == 4 && H == 3 && W == 4) {
        int D_s[] = {0, 0, 2, 2}; 
        int D_e[] = {2, 2, 4, 4}; 
        int H_s[] = {0, 0, 0, 0}; 
        int H_e[] = {3, 3, 3, 3}; 
        int W_s[] = {0, 2, 0, 2}; 
        int W_e[] = {2, 4, 2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    }else if(D == 4 && H == 4 && W == 3) {
        int D_s[] = {0, 0, 2, 2}; 
        int D_e[] = {2, 2, 4, 4}; 
        int H_s[] = {0, 2, 0, 2}; 
        int H_e[] = {2, 4, 2, 4}; 
        int W_s[] = {0, 0, 0, 0}; 
        int W_e[] = {3, 3, 3, 3}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    }else if(D == 4 && H == 4 && W == 4) {
        int D_s[] = {0, 0, 0, 0, 2, 2, 2, 2}; 
        int D_e[] = {2, 2, 2, 2, 4, 4, 4, 4}; 
        int H_s[] = {0, 0, 2, 2, 0, 0, 2, 2}; 
        int H_e[] = {2, 2, 4, 4, 2, 2, 4, 4}; 
        int W_s[] = {0, 2, 0, 2, 0, 2, 0, 2}; 
        int W_e[] = {2, 4, 2, 4, 2, 4, 2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    }else if(D == 5 && H == 1 && W == 1) {
        int D_s[] = {0, 3}; 
        int D_e[] = {3, 5}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {1, 1}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {1, 1}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    }else if(D == 5 && H == 1 && W == 2) {
        int D_s[] = {0, 3}; 
        int D_e[] = {3, 5}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {1, 1}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {2, 2}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    }else if(D == 5 && H == 2 && W == 1) {
        int D_s[] = {0, 3}; 
        int D_e[] = {3, 5}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {2, 2}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {1, 1}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    }else if(D == 5 && H == 2 && W == 2) {
        int D_s[] = {0, 3}; 
        int D_e[] = {3, 5}; 
        int H_s[] = {0, 0}; 
        int H_e[] = {2, 2}; 
        int W_s[] = {0, 0}; 
        int W_e[] = {2, 2}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if(D == 5 && H == 5 && W == 5) {
        int D_s[] = {0, 0, 0, 0, 3, 3, 3, 3}; 
        int D_e[] = {3, 3, 3, 3, 5, 5, 5, 5}; 
        int H_s[] = {0, 0, 3, 3, 0, 0, 3, 3}; 
        int H_e[] = {3, 3, 5, 5, 3, 3, 5, 5}; 
        int W_s[] = {0, 3, 0, 3, 0, 3, 0, 3}; 
        int W_e[] = {3, 5, 3, 5, 3, 5, 3, 5}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 5 && H == 7 && W == 7) {
        int D_s[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3}; 
        int D_e[] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5}; 
        int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6, 0, 0, 0, 3, 3, 3, 6, 6, 6}; 
        int H_e[] = {3, 3, 3, 6, 6, 6, 7, 7, 7, 3, 3, 3, 6, 6, 6, 7, 7, 7}; 
        int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6}; 
        int W_e[] = {3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 7 && H == 7 && W == 7) {
        int D_s[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6}; 
        int D_e[] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7}; 
        int H_s[] = {0, 0, 0, 3, 3, 3, 6, 6, 6, 0, 0, 0, 3, 3, 3, 6, 6, 6, 0, 0, 0, 3, 3, 3, 6, 6, 6}; 
        int H_e[] = {3, 3, 3, 6, 6, 6, 7, 7, 7, 3, 3, 3, 6, 6, 6, 7, 7, 7, 3, 3, 3, 6, 6, 6, 7, 7, 7}; 
        int W_s[] = {0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6}; 
        int W_e[] = {3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 8 && H == 4 && W == 4) {
        int D_s[] = {0, 0, 0, 0, 3, 3, 3, 3, 6, 6, 6, 6}; 
        int D_e[] = {3, 3, 3, 3, 6, 6, 6, 6, 8, 8, 8, 8}; 
        int H_s[] = {0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2}; 
        int H_e[] = {2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4}; 
        int W_s[] = {0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2}; 
        int W_e[] = {2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else if (D == 10 && H == 3 && W == 3) {
        int D_s[] = {0, 3, 6, 9}; 
        int D_e[] = {3, 6, 9, 10}; 
        int H_s[] = {0, 0, 0, 0}; 
        int H_e[] = {3, 3, 3, 3}; 
        int W_s[] = {0, 0, 0, 0}; 
        int W_e[] = {3, 3, 3, 3}; 
        *num_split = sizeof(D_s) / sizeof(D_s[0]); 
        tensorCopy(D_s, H_s, W_s, D_e, H_e, W_e, num_split, D_start, D_end, H_start, H_end, W_start, W_end);
    } else {
      std::cout << D << "x" << H << "x" << W << " not implemented yet." << std::endl;
      exit(-1);
    }
}

#endif //BASE_CONV_LAUNCHERS_H_
