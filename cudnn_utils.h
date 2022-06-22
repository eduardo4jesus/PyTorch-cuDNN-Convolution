#pragma once
#ifndef __MY_CUDNN_UTILS_H__
#define __MY_CUDNN_UTILS_H__

#include <torch/extension.h>
#include <cudnn.h>
#include <iostream>
#include <stdio.h>

/**
 * "desc" must exist where checkCUDNN is called.
 * I do not like this approach, but it is the way to go for now.
 */
#define checkCUDNN(expression)                                             \
  {                                                                        \
    cudnnStatus_t status = (expression);                                   \
    std::ostringstream stream_out;                                         \
    stream_out << "ERROR on line " << __LINE__ << ": "                     \
                << cudnnGetErrorString(status) << " "                      \
                << desc << std::endl;                                      \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, stream_out.str().c_str()); \
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

void initialize_descriptors(const uint B, const uint F, const uint C, 
                            const uint H, const uint W, 
                            const uint KH, const uint KW,
                            const uint OH, const uint OW,
                            c10::ArrayRef<int64_t> &stride,
                            c10::ArrayRef<int64_t> &padding,
                            c10::ArrayRef<int64_t> &dilation,
                            bool channel_first, cudnnDescriptors_t &descriptors,
                            cudnnDataType_t dataType = CUDNN_DATA_FLOAT);

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionFwdAlgoPerf_t &fwdAlgoPert);
std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgoPerf);
std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgoPerf);
std::ostream& operator<<(std::ostream &out, const cudnnDescriptors_t &desc);

#endif