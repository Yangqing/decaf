#include <cublas.h>
#include "util.cuh"

int init_cuda() {
    cudaError err;
    cublasStatus cublas_err;
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    CUDA_RETURN_ON_FAILURE(err);
    cublas_err = cublasInit();
    CUBLAS_RETURN_ON_FAILURE(cublas_err);
    return cudaSuccess;
}
