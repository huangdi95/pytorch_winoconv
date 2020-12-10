#ifndef BASE_CONV_LAUNCHERS_H_
#define BASE_CONV_LAUNCHERS_H_

template <typename T>
void convLauncherStrideOne1x1(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
template <typename T>
void convLauncherStrideOne1x2(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
template <typename T>
void convLauncherStrideOne1x3(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
template <typename T>
void convLauncherStrideOne2x1(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
template <typename T>
void convLauncherStrideOne2x2(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
template <typename T>
void convLauncherStrideOne2x3(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
template <typename T>
void convLauncherStrideOne3x1(const T *input, const T *weight, 
                              T *tmp_input_buffer, T *tmp_weight_buffer, 
                              T *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);
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
                              int C, int B, int H, int W, int K, int pad_h, int pad_w,
                              T *output);

#endif //BASE_CONV_GRAD_LAUNCHERS_H_
