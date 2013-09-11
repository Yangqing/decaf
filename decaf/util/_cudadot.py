"""_cudafuncs is the internal module that actually defines the functions for
cuda.
"""
import ctypes as ct
from decaf import base
import logging
import numpy as np

try:
    _CUDA = ct.CDLL('libcudart.so')
    _CUBLAS = ct.CDLL('libcublas.so')
except OSError:
    _CUDA = ct.CDLL('libcudart.dylib')
    _CUBLAS = ct.CDLL('libcublas.dylib')

c_enum = ct.c_uint

# memcpy flag
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

#####################################################
# Utility functions.
#####################################################

class DecafCudaError(base.DecafError):
    pass


def CUDA_CHECK(error, msg):
    if error:
        _CUDA.cudaDeviceReset()
        raise DecafCudaError('%s (error code %u)' % (msg, error))


def cudaMemcpy(dst, src):
    if isinstance(src, CudaMemory):
        count = src.size()
        src = src.pointer()
        if isinstance(dst, CudaMemory):
            kind = cudaMemcpyDeviceToDevice
            dst = dst.pointer()
        elif isinstance(dst, np.ndarray):
            kind = cudaMemcpyDeviceToHost
            dst = dst.ctypes.data_as(ct.c_void_p)
        else:
            raise ValueError('Unknown dst type.')
    elif isinstance(src, np.ndarray):
        count = src.nbytes
        src = src.ctypes.data_as(ct.c_void_p)
        if isinstance(dst, CudaMemory):
            kind = cudaMemcpyHostToDevice
            dst = dst.pointer()
        elif isinstance(dst, np.ndarray):
            kind = cudaMemcpyHostToHost
            dst = dst.ctypes.data_as(ct.c_void_p)
        else:
            raise ValueError('Unknown dst type.')
    else:
        raise ValueError('Unknown src type.')
    # carry out the memcpy
    CUDA_CHECK(_CUDA.cudaMemcpy(dst, src, ct.c_int(count), ct.c_int(kind)),
               'cudaMemcpy failed.')

#####################################################
# Cublas.
#####################################################

_CUBLAS_HANDLER = ct.c_void_p()
CUDA_CHECK(_CUBLAS.cublasCreate_v2(ct.byref(_CUBLAS_HANDLER)),
           'cublasCreate failed.')

#####################################################
# Cuda Memory.
#####################################################

class CudaMemory(object):
    """A wrapper to avoid losing pointers after allocating them."""
    def __init__(self, size=None):
        """Initializes the object."""
        self._pointer = ct.c_void_p()
        self._size = 0
        if size:
            self.malloc(size)

    def malloc(self, size):
        """Run cudaMalloc to allocate memory on cuda.
        Input:
            size: the number of bytes to allocate.
        """
        self._free()
        CUDA_CHECK(_CUDA.cudaMalloc(ct.byref(self._pointer), ct.c_int(size)),
                   'Memory allocation failed.')
        self._size = size

    def size(self):
        return self._size

    def pointer(self):
        return self._pointer

    def _free(self):
        if self._pointer:
            CUDA_CHECK(_CUDA.cudaFree(self._pointer),
                       'Memory deallocation failed.')
        self._pointer = ct.c_void_p()
        self._size = 0

    def __del__(self):
        self._free()



#####################################################
# Cublas gemm.
#####################################################
def _gemm_f_contiguous(alpha, A, B, out):
    '''A gemm function that uses scipy fblas functions, avoiding matrix copy
    when the input is transposed.
    
    The returned matrix is designed to be F_CONTIGUOUS.
    '''
    if out.shape != (A.shape[0], B.shape[1]):
        raise ValueError("Incorrect output shape.")
    if out.dtype != A.dtype:
        raise ValueError("Incorrect output dtype.")
    if not out.flags.f_contiguous:
        raise ValueError("Output is not f-contiguous.")
    if A.dtype != B.dtype:
        raise TypeError('The data type of the matrices should be the same.')
    if A.dtype == np.float32:
        gemm = _CUBLAS.cublasSgemm_v2
        alpha = ct.c_float(alpha)
        beta = ct.c_float(0)
    elif A.dtype == np.float64:
        gemm = _CUBLAS.cublasDgemm_v2
        alpha = ct.c_double(alpha)
        beta = ct.c_double(0)
    else:
        raise TypeError('Unfit data type.')
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices are not aligned")
    # figure out the dimensions
    m, k = A.shape
    n = B.shape[1]
    if A.flags.c_contiguous:
        lda = A.shape[1]
        A = A.T
        trans_a = CUBLAS_OP_T
    elif A.flags.f_contiguous:
        lda = A.shape[0]
        trans_a = CUBLAS_OP_N
    else:
        raise ValueError('Incorrect matrix flags for A.')
    if B.flags.c_contiguous:
        ldb = B.shape[1]
        B = B.T
        trans_b = CUBLAS_OP_T
    elif B.flags.f_contiguous:
        ldb = B.shape[0]
        trans_b = CUBLAS_OP_N
    else:
        raise ValueError('Incorrect matrix flags for B.')
    # now, let's start copying data and returning the results
    cudaA = CudaMemory(A.nbytes)
    cudaB = CudaMemory(B.nbytes)
    cudaC = CudaMemory(out.nbytes)
    cudaMemcpy(cudaA, A)
    cudaMemcpy(cudaB, B)
    CUDA_CHECK(gemm(
        _CUBLAS_HANDLER, c_enum(trans_a), c_enum(trans_b), ct.c_int(m),
        ct.c_int(n), ct.c_int(k), ct.byref(alpha), cudaA.pointer(),
        ct.c_int(lda), cudaB.pointer(), ct.c_int(ldb), ct.byref(beta),
        cudaC.pointer(), ct.c_int(out.shape[0])),
        'cublasGemm failed.')
    cudaMemcpy(out, cudaC)
    return out

def _gemm_c_contiguous(alpha, A, B, out):
    """A wrapper that computes C_CONTIGUOUS gemm results."""
    _gemm_f_contiguous(alpha, B.T, A.T, out=out.T)
    return out
