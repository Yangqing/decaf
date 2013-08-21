#ifndef _decaf_cuda_util_h
#define _decaf_cuda_util_h

#define CUDA_RETURN_ON_FAILURE(err) if (err != cudaSuccess) return err
#define CUBLAS_RETURN_ON_FAILURE(err) if (err != CUBLAS_STATUS_SUCCESS) return err

extern "C" {

// The function to initialize the cuda
int init_cuda();

} // extern "C"

#endif // _decaf_cuda_util_h

