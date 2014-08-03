
#include "math_defines.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

cudaStream_t CuContext::GetStream() const
{
    cudaStream_t ret;
    cublasGetStream_v2(CublasHandle, &ret);
    return ret;
}
void CuContext::SetStream(cudaStream_t stream)
{
    cublasSetStream_v2(CublasHandle, stream);
}
