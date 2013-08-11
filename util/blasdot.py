# pylint: disable=C0103
"""Efficient dot functions by calling the basic blas functions from scipy."""

import numpy as np
from scipy.linalg import blas
# TODO: maybe add the backward compatibility to old scipy versions.

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
        gemm = blas.sgemm
    elif A.dtype == np.float64:
        gemm = blas.dgemm
    else:
        raise TypeError('Unfit data type.')
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices are not aligned")
    if A.flags.c_contiguous and B.flags.c_contiguous:
        gemm(alpha, a=A.T, b=B.T, trans_a=True, trans_b=True, c=out,
             overwrite_c=True)
    elif A.flags.c_contiguous and B.flags.f_contiguous:
        gemm(alpha, a=A.T, b=B, trans_a=True, c=out, overwrite_c=True)
    elif A.flags.f_contiguous and B.flags.c_contiguous:
        gemm(alpha, a=A, b=B.T, trans_b=True, c=out, overwrite_c=True)
    elif A.flags.f_contiguous and B.flags.f_contiguous:
        gemm(alpha, a=A, b=B, c=out, overwrite_c=True)
    else:
        raise ValueError('Incorrect matrix flags.')
    return out


def _gemm_c_contiguous(alpha, A, B, out):
    """A wrapper that computes C_CONTIGUOUS gemm results."""
    _gemm_f_contiguous(alpha, B.T, A.T, out=out.T)
    return out

def dot(A, B, out=None):
    '''
    a simple wrapper that mimics np.dot (if A and B are both matrices!)
    This function solves the problem that np.dot copies matrices when
    working on transposed matrices.
    Input:
        A, B: two matrices. should be either c-contiguous or f-contiguous
        out: (optional) the output matrix. If it is passed, the matrix should
            have the right shape and should be C_CONTIGUOUS.
    Output:
        out: the output matrix
    Raises:
        TypeError, if the type of matrices is wrong.
    '''
    if out == None:
        out = np.empty((A.shape[0], B.shape[1]), A.dtype, B.dtype)
    # Numpy seems to have bugs dealing with the flags of 1x1 matrices. Thus,
    # if we encounter 1x1 matrices, we manually deal with the calculation.
    if out.size == 1:
        out[:] = np.dot(A.flat, B.flat)
    elif out.flags.f_contiguous:
        out = _gemm_f_contiguous(1.0, A, B, out=out)
    else:
        out = _gemm_c_contiguous(1.0, A, B, out=out)
    return out

def dot_lastdim(A, B, out=None):
    """Performs dot for multi-dimensional matrices A and B, where
    A.shape[-1] = B.shape[0]. The returned matrix should have shape
    A.shape[:-1] + B.shape[1:].

    A and B should both be c-contiguous, otherwise the code will report
    an error.
    """
    if out == None:
        out = np.empty(A.shape[:-1] + B.shape[1:], A.dtype)
    lda = A.size / A.shape[-1]
    dim = A.shape[-1]
    ldb = B.size / B.shape[0]
    # using views
    Aview = A.view()
    Bview = B.view()
    outview = out.view()
    Aview.shape = (lda, dim)
    Bview.shape = (dim, ldb)
    outview.shape = (lda, ldb)
    dot(Aview, Bview, outview)
    return out

def dot_firstdims(A, B, out=None):
    """Performs dot for multi-dimensional matrices A and B, where
    prod(A.shape[:-1]) = prod(B.shape[:-1]), and the result would be
    dot(A.T, B) where A and B are treated as 2-dimensional matrices with shape
    (prod(shape[:-1]), shape[-1]). The returned matrix should have shape
    (A.shape[-1], B.shape[-1]). The code is often encountered in computing the
    gradient in e.g. convolutions.
    
    A and B should both be c-contiguous, otherwise the code will report
    an error.
    """
    if out == None:
        out = np.empty((A.shape[-1], B.shape[-1]), A.dtype)
    lda = A.shape[-1]
    dim = A.size / A.shape[-1]
    ldb = B.shape[-1]
    Aview = A.view()
    Bview = B.view()
    Aview.shape = (dim, lda)
    Bview.shape = (dim, ldb)
    dot(Aview.T, Bview, out)
    return out


