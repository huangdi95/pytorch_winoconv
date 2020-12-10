#ifndef MATH_FUNCTIONS_H_
#define MATH_FUNCTIONS_H_

//#include <boost/math/special_functions/next.hpp>
//#include <boost/random.hpp>

//#include <dmlc/logging.h>
//#include <mxnet/base.h>
//#include <mshadow/base.h>

#include <limits>
#include <fstream>
using std::ofstream;

//// added by zsy
//template <typename T>
//T cpu_max(int size, T* data);
//
//template <typename T>
//T gpu_find_max(const int count, T* data);
//
//template <typename T>
//T gpu_find_min(const int count, T* data);
//
//template <typename T>
//T cpu_min(int size, T* data);
//
//template <typename T>
//void gpu_FloatToFixToFloat_zsy_1(T *data, int bitnum, T max, T min, int data_size, int thread_size, cudaStream_t s);

//added by hd
//template <typename T>
//T FloatToFixToFloat(T data, int bitnum, int shift){
//  const int MAX = 0xffffffff >> (32 - bitnum + 1);
//  const int MIN = int(0xffffffff << (bitnum - 1));
//  shift = -shift;
//  T MASK = 1.0;
//  int result_32;
//  if (shift < 0) {
//    shift = -shift;
//    for (int i = 0; i < shift / 30; ++i) {
//      MASK *= (T)(1 << 30);
//    }
//    MASK *= (T)(1 << (shift % 30));
//    MASK = 1 / MASK;
//  } else {
//    for (int i = 0; i < shift / 30; ++i) {
//      MASK *= (T)(1 << 30);
//    }
//    MASK *= (T)(1 << (shift % 30));
//  }
//  T modify = data * MASK;
//  if (MAX < modify) {
//    result_32 = MAX; 
//  } else if (modify < MIN) {
//    result_32 = MIN; 
//  } else {
//    result_32 = int(modify);
//  }
//  data = T(result_32) / MASK;
//  return data;
//}

//// added by zsy
//template <typename T>
//struct gpu_FloatToFixToFloat {
//  void operator() (T *changed_data, T *data, int bitnum, int shift, int data_size, int thread_size, cudaStream_t s);
//  void operator() (T *changed_data, T *data, int bitnum, float shift, int data_size, int thread_size, cudaStream_t s);
//}

//template <typename T>
//T CountDiff(T *changed_data, T *data, T *data_diff, int data_size, int thread_size, cudaStream_t s);
//
//template <typename T>
//void gpu_FloatToFixToFloat_zsy(T *data, int bitnum, T scale, int data_size, int thread_size, cudaStream_t s);
template <typename T>
void gpu_FloatToFixToFloat_test(T *data, T* changed_data, int bitnum, int shift, float scale, int threadIdxx, int threadIdxy, int threadIdxz, int blockIdxx, int blockIdxy, int blockIdxz);

template <typename T>
void gpu_FloatToFixToFloat_half_test(T *data, T* changed_data, int bitnum, int shift, float scale, int threadIdxx, int threadIdxy, int threadIdxz, int blockIdxx, int blockIdxy, int blockIdxz);

template <typename T>
void gpu_FloatToFixToFloat(T *data, int bitnum, int shift, int threadIdxx, int threadIdxy, int threadIdxz, int blockIdxx, int blockIdxy, int blockIdxz);

template <typename T>
void gpu_FloatToFixToFloatChannel_test(T *data, T* changed_data, int bitnum, const float *shift, const float *scale, int C, int K, int kernel_size);

template <typename T>
void gpu_FloatToFixToFloatChannel_half_test(T *data, T* changed_data, int bitnum, const float *shift, const float *scale, int C, int K, int kernel_size);

template <typename T>
float getShift(T* data, int bitnum, int size);

template <typename T>
void getShift_test(T* data, int bitnum, float *shift, float *scale, int size);

template <typename T>
void getShift_half_test(T* data, int bitnum, float *shift, float *scale, int size);

template <typename T>
void getShift_channel(T* data, int bitnum, float *shift, float *scale, int size, int K, float *blk_vals_c, int *blk_idxs_c, int *blk_num_c);

template <typename T>
void getShift_channel_half(T* data, int bitnum, float *shift, float *scale, int size, int K, float *blk_vals_c, int *blk_idxs_c, int *blk_num_c);

void getM(float *shift, float *scale, float *mean, float alpha, float *mean_out);

template <typename T>
float getDiffBit(T* data_fp, T* data_fix, int size);

template <typename T>
T gpu_absmean(int size, T* data);

//template <typename T>
//void print_tensor(const T* tensor, int size);
#endif  // MATH_FUNCTIONS_H_
