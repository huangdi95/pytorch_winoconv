template <typename Dtype>
void AddGPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);
template <typename Dtype>
void WinoSplit(at::Tensor in_a, at::Tensor out, int block_D, int block_H, int block_W, int output_block_size, int nD, int nH, int nW, bool two_dimension);
template <typename Dtype>
void WinoConcat(at::Tensor in_a, at::Tensor out, int output_block_size, bool two_dimension);
template <typename Dtype>
void DWM(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer);
template <typename Dtype>
void DWM2D(at::Tensor Input, at::Tensor Weight, at::Tensor Output, int stride,
              at::Tensor tmp_input_buffer, at::Tensor tmp_weight_buffer, at::Tensor tmp_product_buffer, at::Tensor tmp_ptr_buffer, int num_split, at::Tensor H_s, at::Tensor H_e, at::Tensor W_s, at::Tensor W_e, at::Tensor tmp_out_buffer);

template <typename Dtype>
void FDWM2D(
  at::Tensor Input,
  at::Tensor Weight,
  at::Tensor Output,
  int stride,
  at::Tensor tmp_weight_buffer,
  int num_split,
  at::Tensor H_s,
  at::Tensor H_e,
  at::Tensor W_s,
  at::Tensor W_e,
  at::Tensor kernel_stride,
  at::Tensor tmp_out_buffer);
