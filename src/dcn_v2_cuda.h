// #ifndef DCN_V2_CUDA
// #define DCN_V2_CUDA

// #ifdef __cplusplus
// extern "C"
// {
// #endif

void dcn_v2_cuda_forward(THCudaTensor *input, THCudaTensor *weight,
                         THCudaTensor *bias, THCudaTensor *ones,
                         THCudaTensor *offset, THCudaTensor *mask,
                         THCudaTensor *output, THCudaTensor *columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group);
void dcn_v2_cuda_backward(THCudaTensor *input, THCudaTensor *weight,
                          THCudaTensor *bias, THCudaTensor *ones,
                          THCudaTensor *offset, THCudaTensor *mask,
                          THCudaTensor *columns,
                          THCudaTensor *grad_input, THCudaTensor *grad_weight,
                          THCudaTensor *grad_bias, THCudaTensor *grad_offset,
                          THCudaTensor *grad_mask, THCudaTensor *grad_output,
                          int kernel_h, int kernel_w,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int deformable_group);

// #ifdef __cplusplus
// }
// #endif

// #endif