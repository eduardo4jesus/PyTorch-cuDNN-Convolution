/**
 * The #include<ATen/cudnn/*.h> needs guards as pointed in
 * https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/cudnn
 */
#include <ATen/cuda/CUDAConfig.h> // for the definition of AT_CUDNN_ENABLED
#if AT_CUDNN_ENABLED()

#include "cudnn_utils.h"
#include <ATen/cudnn/Types.h>      // for getCudnnDataType

void initialize_descriptors(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &output,
                            c10::ArrayRef<int64_t> &stride,
                            c10::ArrayRef<int64_t> &padding,
                            c10::ArrayRef<int64_t> &dilation,
                            cudnnDescriptors_t &desc)
{
  /*****************************************************************************
   * 1. Initializing Descriptors
   ****************************************************************************/
  assert(input.dim() == 4);
  checkCUDNN(cudnnCreateTensorDescriptor(&desc.input));
  checkCUDNN(cudnnSetTensor4dDescriptor(desc.input,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/at::native::getCudnnDataTypeFromScalarType(input.scalar_type()),
                                        /*batch_size=*/input.size(0),
                                        /*channels=*/input.size(1),
                                        /*image_height=*/input.size(2),
                                        /*image_width=*/input.size(3)));

  assert(weight.dim() == 4);
  checkCUDNN(cudnnCreateFilterDescriptor(&desc.weight));
  checkCUDNN(cudnnSetFilter4dDescriptor(desc.weight,
                                        /*dataType=*/at::native::getCudnnDataTypeFromScalarType(weight.scalar_type()),
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/weight.size(0),
                                        /*in_channels=*/weight.size(1),
                                        /*kernel_height=*/weight.size(2),
                                        /*kernel_width=*/weight.size(3)));

  checkCUDNN(cudnnCreateConvolutionDescriptor(&desc.convolution));
  checkCUDNN(cudnnSetConvolution2dDescriptor(desc.convolution,
                                             /*pad_height=*/padding[0],
                                             /*pad_width=*/padding[1],
                                             /*vertical_stride=*/stride[0],
                                             /*horizontal_stride=*/stride[1],
                                             /*dilation_height=*/dilation[0],
                                             /*dilation_width=*/dilation[1],
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/at::native::getCudnnDataTypeFromScalarType(output.scalar_type())));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(desc.convolution,
                                                   desc.input,
                                                   desc.weight,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  assert(batch_size == output.size(0) && channels == output.size(1) &&
    height == output.size(2) && width == output.size(3));

  checkCUDNN(cudnnCreateTensorDescriptor(&desc.output));
  checkCUDNN(cudnnSetTensor4dDescriptor(desc.output,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/at::native::getCudnnDataTypeFromScalarType(output.scalar_type()),
                                        /*batch_size=*/batch_size,
                                        /*channels=*/channels,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

  desc.B = input.size(0); desc.C = input.size(1); desc.H = input.size(1); desc.W = input.size(1);
  desc.F = weight.size(0); desc.KH = weight.size(2); desc.KW = weight.size(3);
  desc.OH = output.size(2); desc.OW = output.size(3);
}

void initialize_descriptors(const uint B, const uint F, const uint C, 
                            const uint H, const uint W, 
                            const uint KH, const uint KW,
                            const uint OH, const uint OW,
                            c10::ArrayRef<int64_t> &stride,
                            c10::ArrayRef<int64_t> &padding,
                            c10::ArrayRef<int64_t> &dilation,
                            cudnnDescriptors_t &desc,
                            cudnnDataType_t dataType)
{
  /*****************************************************************************
   * 1. Initializing Descriptors
   ****************************************************************************/
  checkCUDNN(cudnnCreateTensorDescriptor(&desc.input));
  checkCUDNN(cudnnSetTensor4dDescriptor(desc.input,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/dataType,
                                        /*batch_size=*/B,
                                        /*channels=*/C,
                                        /*image_height=*/H,
                                        /*image_width=*/W));

  checkCUDNN(cudnnCreateFilterDescriptor(&desc.weight));
  checkCUDNN(cudnnSetFilter4dDescriptor(desc.weight,
                                        /*dataType=*/dataType,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/F,
                                        /*in_channels=*/C,
                                        /*kernel_height=*/KH,
                                        /*kernel_width=*/KW));

  checkCUDNN(cudnnCreateConvolutionDescriptor(&desc.convolution));
  checkCUDNN(cudnnSetConvolution2dDescriptor(desc.convolution,
                                             /*pad_height=*/padding[0],
                                             /*pad_width=*/padding[1],
                                             /*vertical_stride=*/stride[0],
                                             /*horizontal_stride=*/stride[1],
                                             /*dilation_height=*/dilation[0],
                                             /*dilation_width=*/dilation[1],
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/dataType));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(desc.convolution,
                                                   desc.input,
                                                   desc.weight,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  assert(batch_size == B && channels == F && height == OH && width == OW);

  checkCUDNN(cudnnCreateTensorDescriptor(&desc.output));
  checkCUDNN(cudnnSetTensor4dDescriptor(desc.output,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/dataType,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/channels,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

  desc.B = B; desc.C = C; desc.H = H; desc.W = W;
  desc.F = F; desc.KH = KH; desc.KW = KW;
  desc.OH = OH; desc.OW = OW;
}

std::ostream &operator<<(std::ostream &out, const cudnnConvolutionFwdAlgo_t &algo)
{
  out << "FWD Algorithm: ";
  switch (algo)
  {
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
    break;
  case CUDNN_CONVOLUTION_FWD_ALGO_COUNT:
    out << "CUDNN_CONVOLUTION_FWD_ALGO_COUNT";
  default:
    std::cerr << "Invalid value FWD Algorithm" << std::endl;
    exit(1);
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const cudnnStatus_t &status) {
  out << "Status: ";
  switch (status)
  {
    case CUDNN_STATUS_SUCCESS:
      out << "CUDNN_STATUS_SUCCESS";
      break;
    case CUDNN_STATUS_NOT_INITIALIZED:
      out << "CUDNN_STATUS_NOT_INITIALIZED";
      break;
    case CUDNN_STATUS_ALLOC_FAILED:
      out << "CUDNN_STATUS_ALLOC_FAILED";
      break;
    case CUDNN_STATUS_BAD_PARAM:
      out << "CUDNN_STATUS_BAD_PARAM";
      break;
    case CUDNN_STATUS_INTERNAL_ERROR:
      out << "CUDNN_STATUS_INTERNAL_ERROR";
      break;
    case CUDNN_STATUS_INVALID_VALUE:
      out << "CUDNN_STATUS_INVALID_VALUE";
      break;
    case CUDNN_STATUS_ARCH_MISMATCH:
      out << "CUDNN_STATUS_ARCH_MISMATCH";
      break;
    case CUDNN_STATUS_MAPPING_ERROR:
      out << "CUDNN_STATUS_MAPPING_ERROR";
      break;
    case CUDNN_STATUS_EXECUTION_FAILED:
      out << "CUDNN_STATUS_EXECUTION_FAILED";
      break;
    case CUDNN_STATUS_NOT_SUPPORTED:
      out << "CUDNN_STATUS_NOT_SUPPORTED";
      break;
    case CUDNN_STATUS_LICENSE_ERROR:
      out << "CUDNN_STATUS_LICENSE_ERROR";
      break;
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      out << "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
      break;
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      out << "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
      break;
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      out << "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
      break;
    case CUDNN_STATUS_VERSION_MISMATCH:
      out << "CUDNN_STATUS_VERSION_MISMATCH";
      break;
    default:
      std::cerr << "Invalid value Status Value" << std::endl;
      exit(1);
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const cudnnDeterminism_t &determinism) {
  out << "Determinism: ";
  switch (determinism)
  {
    case CUDNN_NON_DETERMINISTIC:
      out << "CUDNN_NON_DETERMINISTIC";
      break;
    case CUDNN_DETERMINISTIC:
      out << "CUDNN_DETERMINISTIC";
      break;
    default:
      std::cerr << "Underfined Value: " << static_cast<int>(determinism);
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const cudnnMathType_t &mathType) {
  out << "MathType: ";
  switch (mathType)
  {
  case CUDNN_DEFAULT_MATH:
    out << "CUDNN_DEFAULT_MATH";
    break;
  case CUDNN_TENSOR_OP_MATH:
    out << "CUDNN_TENSOR_OP_MATH";
    break;
  case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
    out << "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
    break;
  case CUDNN_FMA_MATH:
    out << "CUDNN_FMA_MATH";
    break;
  default:
      std::cerr << "Invalid (" << mathType 
      << ")value Algorithm Memory Type" << std::endl;
      exit(1);
  }
  return out;
}

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionFwdAlgoPerf_t &fwdAlgoPert) {
  out << fwdAlgoPert.algo;
  out << "\n\t" << fwdAlgoPert.status;
  out << "\n\tTime: " << fwdAlgoPert.time;
  out << "\n\tMemory: " << fwdAlgoPert.memory;
  out << "\n\t" << fwdAlgoPert.determinism;
  out << "\n\t" << fwdAlgoPert.mathType;
  out << std::endl;
  return out;
}

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgoPerf) {
  out << bwdFilterAlgoPerf.algo;
  out << "\n\t" << bwdFilterAlgoPerf.status;
  out << "\n\tTime: " << bwdFilterAlgoPerf.time;
  out << "\n\tMemory: " << bwdFilterAlgoPerf.memory;
  out << "\n\t" << bwdFilterAlgoPerf.determinism;
  out << "\n\t" << bwdFilterAlgoPerf.mathType;
  out << std::endl;
  return out;
}

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgoPerf) {
  out << bwdDataAlgoPerf.algo;
  out << "\n\t" << bwdDataAlgoPerf.status;
  out << "\n\tTime: " << bwdDataAlgoPerf.time;
  out << "\n\tMemory: " << bwdDataAlgoPerf.memory;
  out << "\n\t" << bwdDataAlgoPerf.determinism;
  out << "\n\t" << bwdDataAlgoPerf.mathType;
  out << std::endl;
  return out;
}

std::ostream& operator<<(std::ostream &out, const cudnnDescriptors_t &desc) {
  out << "Descriptors:";
  out << " input(" << desc.B << ", " << desc.C << ", " << desc.H << ", " << desc.W << ")";
  out << " weight(" << desc.F << ", " << desc.C << ", " << desc.KH << ", " << desc.KW << ")";
  out << " output(" << desc.B << ", " << desc.F << ", " << desc.OH << ", " << desc.OW << ").";
  return out;
}

#endif
