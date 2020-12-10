#ifndef SPLIT_FUNCTIONS_GRAD_H_
#define SPLIT_FUNCTIONS_GRAD_H_
template <typename T>
void splitFeatureGradLauncher(const T *Output_grad, T *Input_grad, int H_start, int H_end, int W_start, int W_end, int B, int H ,int W, int C, int padH, int padW);
template <typename T>
void splitFilterGradLauncher(const T *Output_grad, T *Input_grad, int H_start, int H_end, int W_start, int W_end, int H, int W ,int C, int K);
#endif //SPLIT_FUNCTIONS_GRAD_H_
