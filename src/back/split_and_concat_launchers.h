#ifndef SPLIT_AND_CONCAT_LAUNCHERS_H_ 
#define SPLIT_AND_CONCAT_LAUNCHERS_H_
template <typename T>
void winoSplitLauncher(const T* norm_input, int B, int H, int W, int C, int pad_h, int pad_w, int block_H, int block_W, int nH, int nW, T* wino_input);
template  <typename T>
void winoConcatLauncher(const T* wino_output, int B, int output_H, int output_W, int K, int output_block_H, int output_block_W, T* wino_input);
#endif //SPLIT_AND_CONCAT_LAUNCHERS_H_
