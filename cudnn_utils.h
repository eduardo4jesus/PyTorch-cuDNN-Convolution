#pragma once
#ifndef __MY_CUDNN_UTILS_H__
#define __MY_CUDNN_UTILS_H__

#include <torch/extension.h>
#include <cudnn.h>
#include <iostream>
// #include <string>
// #include <cstdio>
#include <stdio.h>

/**
 * "desc" must exist where checkCUDNN is called.
 * I do not like this approach, but it is the way to go for now.
 */
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS)                      \
    {                                                        \
      std::ostringstream stream_out;                         \
      stream_out << "Error on line " << __LINE__ << ": "     \
                 << cudnnGetErrorString(status) << std::endl \
                 << desc << std::endl;                       \
      std::cerr << stream_out.str();                         \
      TORCH_CHECK(false, stream_out.str().c_str());          \
    }                                                        \
  }

typedef struct _cudnnDescriptors_t_
{
  cudnnTensorDescriptor_t input, output;
  cudnnFilterDescriptor_t weight;
  cudnnConvolutionDescriptor_t convolution;
  int B, C, F, H, W, KH, KW, OH, OW;

  inline virtual ~_cudnnDescriptors_t_()
  {
    cudnnDestroyTensorDescriptor(input);
    cudnnDestroyTensorDescriptor(output);
    cudnnDestroyFilterDescriptor(weight);
    cudnnDestroyConvolutionDescriptor(convolution);
  }
} cudnnDescriptors_t;

void initialize_descriptors(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &output,
                                          c10::ArrayRef<int64_t> &stride,
                                          c10::ArrayRef<int64_t> &padding,
                                          c10::ArrayRef<int64_t> &dilation,
                                          cudnnDescriptors_t &descriptors);

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionFwdAlgoPerf_t &fwdAlgoPert);
std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgoPerf);
std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgoPerf);
std::ostream& operator<<(std::ostream &out, const cudnnDescriptors_t &desc);

#endif