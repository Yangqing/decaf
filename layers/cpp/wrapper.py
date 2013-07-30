"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os

# first, let's import the library
try:
    _cpputil = np.ctypeslib.load_library('libcpputil.so',
            os.path.join(os.path.dirname(__file__)))
except Exception, e:
    raise RuntimeError, "I cannot load libcpputil.so. Please compile first."

################################################################################
# im2col operation
################################################################################
_cpputil.im2col_float.restype = None
_cpputil.im2col_float.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32,
                                                        flags='C'),
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  np.ctypeslib.ndpointer(dtype=np.float32,
                                                        flags='C')]

_cpputil.im2col_double.restype = None
_cpputil.im2col_double.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
                                                         flags='C'),
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=np.float64,
                                                         flags='C')]

def im2col(*args):
    """A wrapper of the im2col function."""
    if args[0].dtype == np.float32:
        return _cpputil.im2col_float(*args)
    elif args[0].dtype == np.float64:
        return _cpputil.im2col_double(*args)
    else:
        raise TypeError('Unsupported type: ' + str(args[0].dtype))

################################################################################
# col2im operation
################################################################################
_cpputil.col2im_float.restype = None
_cpputil.col2im_float.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32,
                                                        flags='C'),
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  np.ctypeslib.ndpointer(dtype=np.float32,
                                                        flags='C')]

_cpputil.col2im_double.restype = None
_cpputil.col2im_double.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
                                                         flags='C'),
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=np.float64,
                                                         flags='C')]

def col2im(*args):
    """A wrapper of the col2im function."""
    if args[0].dtype == np.float32:
        return _cpputil.col2im_float(*args)
    elif args[0].dtype == np.float64:
        return _cpputil.col2im_double(*args)
    else:
        raise TypeError('Unsupported type: ' + str(args[0].dtype))
