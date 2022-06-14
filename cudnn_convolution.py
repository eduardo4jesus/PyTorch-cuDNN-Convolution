from enum import Enum

class CudnnConvFwdAlgo(Enum):
  ## This algorithm expresses the convolution as a matrix product without
  #actually explicitly forming the matrix that holds the input tensor data.
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0

  ## This algorithm expresses convolution as a matrix product without actually
  #explicitly forming the matrix that holds the input tensor data, but still
  #needs some memory workspace to precompute some indices in order to facilitate
  #the implicit construction of the matrix that holds the input tensor data.
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1

  ## This algorithm expresses the convolution as an explicit matrix product. A
  # significant memory workspace is needed to store the matrix that holds the
  # input tensor data.
  CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2

  ## This algorithm expresses the convolution as a direct convolution (for
  #example, without implicitly or explicitly doing a matrix multiplication).
  CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3

  ## This algorithm uses the Fast-Fourier Transform approach to compute the
  #convolution. A significant memory workspace is needed to store intermediate
  #results.
  CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4

  ## This algorithm uses the Fast-Fourier Transform approach but splits the
  #inputs into tiles. A significant memory workspace is needed to store
  #intermediate results but less than CUDNN_CONVOLUTION_FWD_ALGO_FFT for large
  #size images.
  CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5

  ## This algorithm uses the Winograd Transform approach to compute the
  #convolution. A reasonably sized workspace is needed to store intermediate
  #results.
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6

  ## This algorithm uses the Winograd Transform approach to compute the
  #convolution. A significant workspace may be needed to store intermediate
  #results.
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7

  ## Look for the fastest method and try to uses it.
  FASTEST = -1

class CudnnConvBwdFilterAlgo(Enum):
  ## This algorithm expresses the convolution as a sum of matrix products
  #without actually explicitly forming the matrix that holds the input tensor
  #data. The sum is done using the atomic add operation, thus the results are
  #non-deterministic.
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0                 = 0 # /* non-deterministic */

  ## This algorithm expresses the convolution as a matrix product without
  #actually explicitly forming the matrix that holds the input tensor data. The
  #results are deterministic.
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1                 = 1

  ##This algorithm uses the Fast-Fourier Transform approach to compute the
  #convolution. A significant workspace is needed to store intermediate results.
  #The results are deterministic.
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT               = 2

  ##This algorithm is similar to CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 but uses
  #some small workspace to precompute some indices. The results are also
  #non-deterministic.
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3                 = 3 # /* non-deterministic */

  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD          = 4 # /* not implemented */

  ## This algorithm uses the Winograd Transform approach to compute the
  #convolution. A significant workspace may be needed to store intermediate
  #results. The results are deterministic.
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5

  ## This algorithm uses the Fast-Fourier Transform approach to compute the
  #convolution but splits the input tensor into tiles. A significant workspace
  #may be needed to store intermediate results. The results are deterministic.
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING        = 6

  ## Look for the fastest method and try to uses it.
  FASTEST = -1
  pass

class CudnnConvBwdDataAlgo(Enum):
  ## This algorithm expresses the convolution as a sum of matrix products
  #without actually explicitly forming the matrix that holds the input tensor
  #data. The sum is done using the atomic add operation, thus the results are
  #non-deterministic.
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                 = 0 # /* non-deterministic */

  ## This algorithm expresses the convolution as a matrix product without
  #actually explicitly forming the matrix that holds the input tensor data. The
  #results are deterministic.
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                 = 1

  ## This algorithm uses a Fast-Fourier Transform approach to compute the
  #convolution. A significant memory workspace is needed to store intermediate
  #results. The results are deterministic.
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT               = 2

  ## This algorithm uses the Fast-Fourier Transform approach but splits the
  #inputs into tiles. A significant memory workspace is needed to store
  #intermediate results but less than CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT for
  #large size images. The results are deterministic.
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING        = 3

  ## This algorithm uses the Winograd Transform approach to compute the
  #convolution. A reasonably sized workspace is needed to store intermediate
  #results. The results are deterministic.
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD          = 4
  
  ## This algorithm uses the Winograd Transform approach to compute the
  #convolution. A significant workspace may be needed to store intermediate
  #results. The results are deterministic.
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5

  ## Look for the fastest method and try to uses it.
  FASTEST = -1