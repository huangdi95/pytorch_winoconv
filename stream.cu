/*************************************************************************
    > Author: Huang Di
    > Mail: hd232508@163.com 
    > Created Time: Sun 14 Mar 2021 04:14:01 PM CST
 ************************************************************************/
template <>
void convLauncherStrideOneLarge2D2<float>(const float *input, const float *weight,
                              float *tmp_input_buffer, float *tmp_weight_buffer,
                              float *tmp_product_buffer, const int64_t *tmp_ptr_buffer,
                              int B, int H, int W, int C, int K,
                              int kernel_H, int kernel_W, int pad_h, int pad_w,
                              float *output,
                              int num_split, int *H_start_gpu, int *H_end_gpu, int *W_start_gpu, int *W_end_gpu, float *tmp_out_buffer)
{
//  using std::chrono::high_resolution_clock;
//  using std::chrono:://duration_cast;
//  using std::chrono:://duration;
//  using std::chrono::milliseconds;
  //auto tt1 = high_resolution_clock::now();
//    //cudaEvent_t t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16;
//    float elapsedTime;

    //cudaEventCreate(&t1);
    //cudaEventRecord(t1, 0);
    int output_H = (H + 2 * pad_h - kernel_H) / 1 + 1;
    int output_W = (W + 2 * pad_w - kernel_W) / 1 + 1;
    int nH = (output_H + 1) / 2;
    int nW = (output_W + 1) / 2;

//////////////////// a large LUT ///////////////////////
    int num_split2;
    int *H_start = nullptr;
    int *W_start = nullptr;
    int *H_end = nullptr;
    int *W_end = nullptr;
    splitControl2D(kernel_H, kernel_W, &num_split2, &H_start, &H_end, &W_start, &W_end); 

    //cudaEventCreate(&t2);
    //cudaEventRecord(t2,0);
    //cudaEventSynchronize(t2);
  //auto tt2 = high_resolution_clock::now();
    //duration<double, std::milli> //ms_double = tt2 - tt1;
    //std::cout << "splitcontrol cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t1, t2);
    //printf("splitcontrol time : %f ms\n" ,elapsedTime);

    int *kernel_stride = new int[num_split]();
    for(int i = 1; i < num_split; i++) {
        kernel_stride[i] = (H_end[i-1] - H_start[i-1] + 1) * (W_end[i-1] - W_start[i-1] + 1) + kernel_stride[i-1];
//        cout << kernel_stride[i] << endl;
    }

//    cout << "kernel_size: " << kernel_size << endl;

    //cudaEventCreate(&t3);
    //cudaEventRecord(t3,0);
    //cudaEventSynchronize(t3);
  //auto tt3 = high_resolution_clock::now();
    //ms_double = tt3 - tt2;
    //std::cout << "kernel_stride cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t2, t3);
    //printf("kernel_stride time: %f ms\n" ,elapsedTime);

//    int *H_start_gpu = nullptr;
//    int *W_start_gpu = nullptr;
//    int *H_end_gpu = nullptr;
//    int *W_end_gpu = nullptr;
    int *kernel_stride_gpu = nullptr;
//    cudaMalloc((void**)&H_start_gpu, num_split*sizeof(int));
//    cudaMalloc((void**)&H_end_gpu, num_split*sizeof(int));
//    cudaMalloc((void**)&W_start_gpu, num_split*sizeof(int));
//    cudaMalloc((void**)&W_end_gpu, num_split*sizeof(int));
    cudaMalloc((void**)&kernel_stride_gpu, num_split*sizeof(int));
//    cudaMemcpy(H_start_gpu, H_start, num_split*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(W_start_gpu, W_start, num_split*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(H_end_gpu, H_end, num_split*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(W_end_gpu, W_end, num_split*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_stride_gpu, kernel_stride, num_split*sizeof(int), cudaMemcpyHostToDevice);

    //cudaEventCreate(&t4);
    //cudaEventRecord(t4,0);
    //cudaEventSynchronize(t4);
  //auto tt4 = high_resolution_clock::now();
    //ms_double = tt4 - tt3;
    //std::cout << "malloc cpy cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t3, t4);
    //printf("malloc cpy time: %f ms\n" ,elapsedTime);

//    dim3 bDim1(C, 1, 1);
//    dim3 gDim1(nH*nW, B, num_split);
//    int *time = nullptr;
////    cudaMalloc((void**)&time, num_split**nH*nW*B*C*sizeof(int));
//    cudaMalloc((void**)&time, 9*sizeof(int));
//    inputNorm2WinoTransform2D <float> <<<gDim1, bDim1>>> (input, tmp_input_buffer, kernel_stride_gpu, H_start_gpu, H_end_gpu, W_start_gpu, W_end_gpu, nH, nW, B, H, W, C, pad_h, pad_w, time);
//    int time_host[9];
//    cudaMemcpy(time_host, time, 9*sizeof(int), cudaMemcpyDeviceToHost);
//    for(int i = 0; i < 9; i++) {
//        fprintf(stdout, "%d:%ld=%f(ms)\n", i,time_host[i], ((float)(time_host[i])/1620000000.0f)*1000.0);
//    }
    int kernel_size = 0;
    int batch = 0;
    int N;
//    int s = num_split - 1;
    int step = 3;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    for(int s = 0; s < 9; s+=step) {
    at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(myStream);
    cout << myStream << endl;
    kernel_size += batch;
    batch = 0;
    for(int i = s; i < s + step; i++) {
        batch += (H_end[i] - H_start[i] + 1) * (W_end[i] - W_start[i] + 1);
    }
//    int batch = (H_end[8] - H_start[8] + 1) * (W_end[8] - W_start[8] + 1) + kernel_stride[8];

    N = C * nH * nW * B * step;
    cout << N << endl;
    cout << kernel_size << endl;
    cout << batch << endl;
    cout << s << endl;
    int *time = nullptr;
//    cudaMalloc((void**)&time, num_split**nH*nW*B*C*sizeof(int));
    cudaMalloc((void**)&time, 9*sizeof(int));
    inputNorm2WinoTransform2D2 <float> <<<(N - 1 + 256) / 256, 256, 0, myStream>>> (input, tmp_input_buffer, kernel_stride_gpu+s, H_start_gpu+s, H_end_gpu+s, W_start_gpu+s, W_end_gpu+s, nH, nW, B, H, W, C, pad_h, pad_w, time, N);
    int time_host[9];
    cudaMemcpy(time_host, time, 9*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 9; i++) {
        fprintf(stdout, "%d:%ld=%f(ms)\n", i,time_host[i], ((float)(time_host[i])/1620000000.0f)*1000.0);
    }

    //cudaEventCreate(&t5);
    //cudaEventRecord(t5,0);
    //cudaEventSynchronize(t5);
  //auto tt5 = high_resolution_clock::now();
    //ms_double = tt5 - tt4;
    //std::cout << "input trans cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t4, t5);
    //printf("input trans time : %f ms\n" ,elapsedTime);

    dim3 bDim2(K, 1, 1);
    dim3 gDim2(C, step, 1);
    wNorm2WinoTransform2D <float> <<<gDim2, bDim2, 0, myStream>>> (weight, tmp_weight_buffer, kernel_stride_gpu+s, H_start_gpu+s, H_end_gpu+s, W_start_gpu+s, W_end_gpu+s, kernel_H, kernel_W, C, K);
    //cudaEventCreate(&t6);
    //cudaEventRecord(t6,0);
    //cudaEventSynchronize(t6);
  //auto tt6 = high_resolution_clock::now();
    //ms_double = tt6 - tt5;
    //std::cout << "w trans cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t5, t6);
    //printf("w trans time : %f ms\n" ,elapsedTime);

    const float** Input_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + kernel_size*3);
    const float** Weight_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + kernel_size*3 + batch);
    float** tmp_product_ptrs_gpu_ = (float **)(tmp_ptr_buffer + kernel_size*3 + 2*batch);

    dim3 bDim3(batch, 1, 1);
    dim3 gDim3(1, 1, 1);
//    forwardAssign2D <float> <<<gDim3, bDim3>>> (tmp_input_buffer+kernel_size*B*nH*nW*C, tmp_weight_buffer+kernel_size*C*K, tmp_product_buffer+kernel_size*B*nH*nW*K, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);
    forwardAssign2D <float> <<<gDim3, bDim3>>> (tmp_input_buffer+kernel_size*B*nH*nW*C, tmp_weight_buffer_kernel_size*K*C, tmp_product_buffer+kernel_size*nH*nW*B*K, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);

    float one = 1;
    float zero = 0;
  
    //cudaEventCreate(&t7);
    //cudaEventRecord(t7,0);
    //cudaEventSynchronize(t7);
  //auto tt7 = high_resolution_clock::now();
    //ms_double = tt7 - tt6;
    //std::cout << "forwardassign cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t6, t7);
    //printf("forwardassign time : %f ms\n" ,elapsedTime);

//    cublasHandle_t handle;
//    handle = ;

    //cudaEventCreate(&t14);
    //cudaEventRecord(t14,0);
    //cudaEventSynchronize(t14);
  //auto tt14 = high_resolution_clock::now();
    //ms_double = tt14 - tt7;
    //std::cout << "init handle cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t7, t14);
    //printf("init handle time : %f ms\n" ,elapsedTime);

//    cublasCreate(&handle);

    //cudaEventCreate(&t15);
    //cudaEventRecord(t15,0);
    //cudaEventSynchronize(t15);
  //auto tt15 = high_resolution_clock::now();
    //ms_double = tt15 - tt14;
    //std::cout << "create handle cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t14, t15);
    //printf("create handle time : %f ms\n" ,elapsedTime);
    cublasSetStream(handle, myStream);
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        &one,
        Weight_ptrs_gpu_, K,
        Input_ptrs_gpu_, C,
        &zero, tmp_product_ptrs_gpu_, K, batch);

    //cudaEventCreate(&t8);
    //cudaEventRecord(t8,0);
    //cudaEventSynchronize(t8);
  //auto tt8 = high_resolution_clock::now();
    //ms_double = tt8 - tt15;
    //std::cout << "gemm cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t15, t8);
    //printf("gemm time : %f ms\n" ,elapsedTime);

//    dim3 bDim4(K, 1, 1);
//    dim3 gDim4(nH*nW, B, step);
//    float *tmp_output = nullptr;
//    cudaMalloc((void**)&tmp_output, num_split*B*output_H*output_W*K*sizeof(float));

    //cudaEventCreate(&t16);
    //cudaEventRecord(t16, 0);
    //cudaEventSynchronize(t16);
  //auto tt16 = high_resolution_clock::now();
    //ms_double = tt16 - tt8;
    //std::cout << "out trans cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t8, t16);
    //printf("out trans time : %f ms\n" ,elapsedTime);

    N = step*B*nH*nW*K;
    outputWino2NormTransform2D <float> <<<(N - 1 + 256) / 256, 256, 0, myStream>>> (tmp_product_buffer, tmp_out_buffer+s*B*output_H*output_W*K, kernel_stride_gpu+s, H_start_gpu+s, H_end_gpu+s, W_start_gpu+s, W_end_gpu+s, B, output_H, output_W, K, kernel_size, N);
//    outputWino2NormTransform2D <float> <<<(N - 1 + 256) / 256, 256>>> (tmp_product_buffer, tmp_out_buffer+s*B*output_H*output_W*K, kernel_stride_gpu+s, H_start_gpu+s, H_end_gpu+s, W_start_gpu+s, W_end_gpu+s, B, output_H, output_W, K, kernel_size, s, N);
//    outputWino2NormTransform2D <float> <<<(N - 1 + 256) / 256, 256>>><<<gDim4, bDim4>>> (tmp_product_buffer, tmp_out_buffer+s*B*output_H*output_W*K, kernel_stride_gpu+s, H_start_gpu+s, H_end_gpu+s, W_start_gpu+s, W_end_gpu+s, B, output_H, output_W, K, kernel_size, s);

    }
//    at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
//    at::cuda::setCurrentCUDAStream(defaultStream);

    //cudaEventCreate(&t9);
    //cudaEventRecord(t9,0);
    //cudaEventSynchronize(t9);
  //auto tt9 = high_resolution_clock::now();
    //ms_double = tt9 - tt16;
    //std::cout << "out trans cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t16, t9);
    //printf("out trans time : %f ms\n" ,elapsedTime);

//    dim3 bDim5(K, 1, 1);
//    dim3 gDim5(output_H*output_W, B, 1);
    N = B*output_H*output_W*K;
    outputAggregate2D<float> <<<(N - 1 + 256) / 256, 256>>> (tmp_out_buffer, output, num_split, B, output_H, output_W, K, N);
//    outputAggregate2D<float> <<<gDim5, bDim5>>> (tmp_out_buffer, output, num_split, B, output_H, output_W, K);
    //cudaEventCreate(&t10);
    //cudaEventRecord(t10,0);
    //cudaEventSynchronize(t10);
  //auto tt10 = high_resolution_clock::now();
    //ms_double = tt10 - tt9;
    //std::cout << "aggregate cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t9, t10);
    //printf("aggregate time : %f ms\n" ,elapsedTime);


//    cublasDestroy(handle);
    //cudaEventCreate(&t11);
    //cudaEventRecord(t11,0);
    //cudaEventSynchronize(t11);
  //auto tt11 = high_resolution_clock::now();
    //ms_double = tt11 - tt10;
    //std::cout << "cublasDestroy cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t10, t11);
    //printf("cublasDestroy time: %f ms\n" ,elapsedTime);
//    cudaFree(H_start_gpu);
//    cudaFree(W_start_gpu);
//    cudaFree(H_end_gpu);
//    cudaFree(W_end_gpu);
    cudaFree(kernel_stride_gpu);
//    cudaFree(tmp_output);

    //cudaEventCreate(&t12);
    //cudaEventRecord(t12,0);
    //cudaEventSynchronize(t12);
  //auto tt12 = high_resolution_clock::now();
    //ms_double = tt12 - tt11;
    //std::cout << "free cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t11, t12);
    //printf("free time: %f ms\n" ,elapsedTime);

    delete[] H_start;
    delete[] W_start;
    delete[] H_end;
    delete[] W_end;
    delete[] kernel_stride;
    //cudaEventCreate(&t13);
    //cudaEventRecord(t13,0);
    //cudaEventSynchronize(t13);
  //auto tt13 = high_resolution_clock::now();
    //ms_double = tt13 - tt12;
    //std::cout << "delete cpu: "<< //ms_double.count() << " ms" << endl;

    //cudaEventElapsedTime(&elapsedTime, t12, t13);
    //printf("delete time: %f ms\n" ,elapsedTime);

}
