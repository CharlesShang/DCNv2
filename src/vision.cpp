
#include "dcn_v2.h"
#include <torch/script.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
  m.def("dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward,
        "dcn_v2_psroi_pooling_forward");
  m.def("dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward,
        "dcn_v2_psroi_pooling_backward");
}

inline at::Tensor dcn_v2(const at::Tensor &input, const at::Tensor &weight,
                         const at::Tensor &bias, const at::Tensor &offset,
                         const at::Tensor &mask, const int64_t kernel_h,
                         const int64_t kernel_w, const int64_t stride_h,
                         const int64_t stride_w, const int64_t pad_h,
                         const int64_t pad_w, const int64_t dilation_h,
                         const int64_t dilation_w,
                         const int64_t deformable_group) {
  return dcn_v2_forward(input, weight, bias, offset, mask, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dilation_h,
                        dilation_w, deformable_group);
}
static auto registry = torch::RegisterOperators().op("custom::dcn_v2", &dcn_v2);
