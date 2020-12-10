#ifndef SPLIT_FUNCTIONS_H_
#define SPLIT_FUNCTIONS_H_
template<typename T>
void splitFeatureLauncher(const T *Input, T *Splitout, int H_start, int H_end, int W_start, int W_end, int B, int H ,int W, int C, int padH, int padW);
template<typename T>
void splitFilterLauncher(const T *Input, T *Splitout, int H_start, int H_end, int W_start, int W_end, int H, int W ,int C, int K);
template<typename T>
void addFeatureLauncher(T *Result, T *Input1, T *Input2, int B, int H, int W ,int C);
template<typename T>
void zeroInit(T *grad, int size);
#endif //SPLIT_FUNCTIONS_H_
