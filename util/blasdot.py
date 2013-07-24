import numpy as np
from scipy import linalg

def gemm_f_contiguous(alpha, A, B, out=None):
    '''A gemm function that uses scipy fblas functions, avoiding matrix copy
    when the input is transposed.
    
    The returned matrix is designed to be F_CONTIGUOUS.
    '''
    scipy_gemm = linalg.get_blas_funcs('gemm', arrays=(A,B))
    if A.dtype != B.dtype:
        raise ValueError('The data type of the matrices should be the same.')
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices are not aligned")
    if out is None:
        out = np.empty((A.shape[0], B.shape[1]), A.dtype, order="F")
    elif out.shape != (A.shape[0], B.shape[1]) or out.dtype != A.dtype or \
            not out.flags.f_contiguous:
        raise ValueError("Incorrect output data type.")
    if A.flags.c_contiguous and B.flags.c_contiguous:
        scipy_gemm(alpha, a=A.T, b=B.T, trans_a=True, trans_b=True,
                   c=out, overwrite_c=True)
    elif A.flags.c_contiguous and B.flags.f_contiguous:
        scipy_gemm(alpha, a=A.T, b=B, trans_a=True, c=out, overwrite_c=True)
    elif A.flags.f_contiguous and B.flags.c_contiguous:
        scipy_gemm(alpha, a=A, b=B.T, trans_b=True, c=out, overwrite_c=True)
    elif A.flags.f_contiguous and B.flags.f_contiguous:
        scipy_gemm(alpha, a=A, b=B, c=out, overwrite_c=True)
    else:
        raise ValueError('Incorrect matrix flags.')
    return out


def gemm_c_contiguous(alpha, A, B, out=None):
    """A wrapper that computes C_CONTIGUOUS gemm results."""
    if out is None:
        return gemm_f_contiguous(alpha, B.T, A.T).T
    else:
        gemm_f_contiguous(alpha, B.T, A.T, out=out.T)
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
    return gemm_c_contiguous(1.0, A, B, out=out)

