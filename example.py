import torch
from torch.utils.cpp_extension import load
from cudnn_convolution import CudnnConvFwdAlgo, CudnnConvBwdFilterAlgo, CudnnConvBwdDataAlgo

# load the PyTorch extension
cudnn_convolution = load(
  name="cudnn_convolution",
  sources=["cudnn_convolution.cpp", "cudnn_utils.cpp"],
  extra_ldflags = ["-lcudnn"],
  with_cuda=True,
  verbose=True
)
print("Compiled and Loaded!")


# create dummy input, convolutional weights and bias
B, F, C = 8, 32, 3
N, K, O = 32, 5, 28
input  = torch.zeros(B, C, N, N).to('cuda')
weight = torch.zeros(F, C, K, K).to('cuda')
output = torch.zeros(B, F, O, O).to('cuda')
# print(input.dtype, weight.dtype, output.dtype)

stride   = (1, 1)
padding  = (0, 0)
dilation = (1, 1)
groups   = 1

# compute the result of convolution
output = cudnn_convolution.convolution(
  # CudnnConvFwdAlgo.FASTEST.value,
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM.value,
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM.value, 
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT.value, 
  input, weight, output, stride, padding, dilation, groups, True)

print("Done!")
# # create dummy gradient w.r.t. the output
# grad_output = torch.zeros(128, 64, 14, 14).to('cuda')

# # compute the gradient w.r.t. the weights and input
# grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride, padding, dilation, groups, False, False, False)
# grad_input  = cudnn_convolution.convolution_backward_input(input.shape, weight, grad_output, stride, padding, dilation, groups, False, False, False)

# print(grad_weight.shape)
# print(grad_input.shape)