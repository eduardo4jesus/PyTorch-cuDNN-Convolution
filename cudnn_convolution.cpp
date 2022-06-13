/**
 * The #include<ATen/cudnn/*.h> needs guards as pointed in
 * https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/cudnn
 */
#include <ATen/cuda/CUDAConfig.h> // for the definition of AT_CUDNN_ENABLED
#if AT_CUDNN_ENABLED()

#include <cudnn.h>
#include <torch/extension.h>
#include <ATen/cudnn/Handle.h>     // for getCudnnHandle
#include <ATen/cudnn/Types.h>      // for getCudnnDataType
#include <ATen/native/ConvUtils.h> // for cudnn_conv_suggest_memory_format
// #include <vector>
// #include <ATen/NativeFunctions.h>
// #include <ATen/Config.h>

/*
PyTorch extension enabling direct access to the following cuDNN-accelerated C++ functions
that are included in PyTorch:

    - cudnn_convolution
    - cudnn_convolution_backward_weight
    - cudnn_convolution_backward_input

The functions defined here can be called from Python in replacement of
torch.nn.conv2d, torch.nn.grad.conv2d_weight and torch.nn.grad.conv2d_input,
and run significantly faster. See 'example.py' for how these functions
are called.

Adapted from code posted by goldsborough/conv.cu:
https://gist.github.com/eduardo4jesus/33ef6d8696e8af70a3046e9f364a65f8#file-conv-cu
*/

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

typedef struct
{
  cudnnTensorDescriptor_t idesc, odesc;
  cudnnFilterDescriptor_t wdesc;
  cudnnConvolutionDescriptor_t cdesc;
} cudnnConvArgs;

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionFwdAlgoPerf_t &fwdAlgoPert);

at::Tensor convolution(const int fwdAlgo, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &output,
                       c10::ArrayRef<int64_t> stride, c10::ArrayRef<int64_t> padding, c10::ArrayRef<int64_t> dilation,
                       int64_t groups, bool benchmark, bool deterministic, bool verbose)
{
  const cudnnHandle_t cudnn = at::native::getCudnnHandle();

  cudnnConvArgs args;
  assert(input.dim() == 4);
  checkCUDNN(cudnnCreateTensorDescriptor(&args.idesc));
  checkCUDNN(cudnnSetTensor4dDescriptor(args.idesc,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/at::native::getCudnnDataTypeFromScalarType(input.scalar_type()),
                                        /*batch_size=*/input.size(0),
                                        /*channels=*/input.size(1),
                                        /*image_height=*/input.size(2),
                                        /*image_width=*/input.size(3)));

  assert(weight.dim() == 4);
  checkCUDNN(cudnnCreateFilterDescriptor(&args.wdesc));
  checkCUDNN(cudnnSetFilter4dDescriptor(args.wdesc,
                                        /*dataType=*/at::native::getCudnnDataTypeFromScalarType(weight.scalar_type()),
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/weight.size(0),
                                        /*in_channels=*/weight.size(1),
                                        /*kernel_height=*/weight.size(2),
                                        /*kernel_width=*/weight.size(3)));

  checkCUDNN(cudnnCreateConvolutionDescriptor(&args.cdesc));
  checkCUDNN(cudnnSetConvolution2dDescriptor(args.cdesc,
                                             /*pad_height=*/padding[0],
                                             /*pad_width=*/padding[1],
                                             /*vertical_stride=*/stride[0],
                                             /*horizontal_stride=*/stride[1],
                                             /*dilation_height=*/dilation[0],
                                             /*dilation_width=*/dilation[1],
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/at::native::getCudnnDataTypeFromScalarType(output.scalar_type())));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(args.cdesc,
                                                   args.idesc,
                                                   args.wdesc,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  assert(batch_size == output.size(0) && channels == output.size(1) &&
    height == output.size(2) && width == output.size(3));

  checkCUDNN(cudnnCreateTensorDescriptor(&args.odesc));
  checkCUDNN(cudnnSetTensor4dDescriptor(args.odesc,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/at::native::getCudnnDataTypeFromScalarType(output.scalar_type()),
                                        /*batch_size=*/batch_size,
                                        /*channels=*/channels,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

  cudnnConvolutionFwdAlgoPerf_t convolution_algorithm[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  int returnedAlgoCount;

  /**
   * TODO: I frequently get segmentation fault when finding the convolution
   * algorithms. I am not sure how to fix it.
   */
  if (fwdAlgo == -1) {
    std::cout << "Trying all" << std::endl;
    checkCUDNN(
      cudnnFindConvolutionForwardAlgorithm(cudnn,
                                          args.idesc,
                                          args.wdesc,
                                          args.cdesc,
                                          args.odesc,
                                          /*requestedAlgoCount*/CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                          &returnedAlgoCount,
                                          convolution_algorithm));
    if (verbose)
      for (int i=0; i<returnedAlgoCount; i++)
        std::cout << convolution_algorithm[i] << std::endl;
  } else {
    convolution_algorithm[0].algo = static_cast<cudnnConvolutionFwdAlgo_t>(fwdAlgo);
    convolution_algorithm[0].status = static_cast<cudnnStatus_t>(0); 
    convolution_algorithm[0].time = -1;
    convolution_algorithm[0].memory = 0;
    convolution_algorithm[0].mathType = static_cast<cudnnMathType_t>(0);
    if (verbose) {
      std::cout << "Attempt with defined Algo:" << std::endl;
      std::cout << convolution_algorithm[0] << std::endl;
    }
  }

  if (verbose)
    std::cout << "Allocating Workspace" << std::endl;

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     args.idesc,
                                                     args.wdesc,
                                                     args.cdesc,
                                                     args.odesc,
                                                     convolution_algorithm[0].algo,
                                                     &workspace_bytes));

  if (verbose)
    std::cout << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  const float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     args.idesc,
                                     input.data_ptr(),
                                     args.wdesc,
                                     weight.data_ptr(),
                                     args.cdesc,
                                     convolution_algorithm[0].algo,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     args.odesc,
                                     output.data_ptr()));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds{0};
  cudaEventElapsedTime(&milliseconds, start, stop);
  if (verbose)
    std::cout << "Elapsed Time: " << milliseconds << " ms" << std::endl;

  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(args.idesc);
  cudnnDestroyTensorDescriptor(args.odesc);
  cudnnDestroyFilterDescriptor(args.wdesc);
  cudnnDestroyConvolutionDescriptor(args.cdesc);

  return output;
}

// at::Tensor convolution_backward_weight(
//     const at::Tensor& input,
//     c10::ArrayRef<int64_t> weight_size,
//     const at::Tensor& grad_output,
//     c10::ArrayRef<int64_t> stride,
//     c10::ArrayRef<int64_t> padding,
//     c10::ArrayRef<int64_t> dilation,
//     int64_t groups,
//     bool benchmark,
//     bool deterministic,
//     bool allow_tf32) {

//     return at::cudnn_convolution_backward_weight(
//         weight_size,
//         grad_output,
//         input,
//         padding,
//         stride,
//         dilation,
//         groups,
//         benchmark,
//         deterministic,
//         allow_tf32);
// }

// at::Tensor convolution_backward_input(
//     c10::ArrayRef<int64_t> input_size,
//     const at::Tensor& weight,
//     const at::Tensor& grad_output,
//     c10::ArrayRef<int64_t> stride,
//     c10::ArrayRef<int64_t> padding,
//     c10::ArrayRef<int64_t> dilation,
//     int64_t groups,
//     bool benchmark,
//     bool deterministic,
//     bool allow_tf32) {

//     return at::cudnn_convolution_backward_input(
//         input_size,
//         grad_output,
//         weight,
//         padding,
//         stride,
//         dilation,
//         groups,
//         benchmark,
//         deterministic,
//         allow_tf32);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("convolution", &convolution, "convolution");
  // m.def("convolution_backward_weight", &convolution_backward_weight, "convolution backward weight");
  // m.def("convolution_backward_input", &convolution_backward_input, "convolution backward input");
}

std::ostream& operator<<(std::ostream &out, const cudnnConvolutionFwdAlgoPerf_t &fwdAlgoPert) {

  out << "Algorithm: ";

  switch (fwdAlgoPert.algo) {
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

  out << "\n\tStatus: ";

  switch (fwdAlgoPert.status)
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
      std::cerr << "Invalid value FWD Algorithm Status" << std::endl;
      exit(1);
  }

  out << "\n\tTime: " << fwdAlgoPert.time;
  out << "\n\tMemory: " << fwdAlgoPert.memory;
  out << "\n\tMathType: ";

  switch (fwdAlgoPert.mathType)
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
      std::cerr << "Invalid (" << fwdAlgoPert.mathType 
      << ")value FWD Algorithm Memory Type" << std::endl;
      exit(1);
  }  

  out << std::endl;

  return out;
}

#endif
