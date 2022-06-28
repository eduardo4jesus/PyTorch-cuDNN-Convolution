import torch
from torch.utils.cpp_extension import load
from cudnn_convolution import *

B, F, C = 256, 512, 128
N, K, O = 32, 3, 32
padding = 1

# output = cudnn_convolution_fwd(
#   CudnnConvFwdAlgo.FASTEST,
#   # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT,
#   B, F, C, N, K, O, padding
# )

output = cudnn_find_convolution_fwd_algo_(
  B, F, C, N, N, K, K, O, O, padding, channel_first=True, verbose=True
)

output = cudnn_find_convolution_fwd_algo_(
  B, F, C, N, N, K, K, O, O, padding, channel_first=False, verbose=True
)

# print(output)
print("Done!")
# # create dummy gradient w.r.t. the output
# grad_output = torch.zeros(128, 64, 14, 14).to('cuda')

# # compute the gradient w.r.t. the weights and input
# grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride, padding, dilation, groups, False, False, False)
# grad_input  = cudnn_convolution.convolution_backward_input(input.shape, weight, grad_output, stride, padding, dilation, groups, False, False, False)

# print(grad_weight.shape)
# print(grad_input.shape)
