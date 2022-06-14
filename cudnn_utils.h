#pragma once
#ifndef __MY_CUDNN_UTILS_H__
#define __MY_CUDNN_UTILS_H__

#include <torch/extension.h>
#include <cudnn.h>
#include <iostream>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS)                      \
    {                                                        \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionFwdAlgoPerf_t &fwdAlgoPert);
std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgoPerf);
std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgoPerf);

typedef struct _cudnnDescriptors_t_
{
  cudnnTensorDescriptor_t input, output;
  cudnnFilterDescriptor_t weight;
  cudnnConvolutionDescriptor_t convolution;

  virtual ~_cudnnDescriptors_t_()
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

#endif