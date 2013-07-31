"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os

# first, let's import the library
try:
    _cpp = np.ctypeslib.load_library('libcpputil.so',
            os.path.join(os.path.dirname(__file__)))
except Exception as error:
    raise error

################################################################################
# im2col and col2im operation
################################################################################
_cpp.im2col_sc_float.restype = \
_cpp.im2col_mc_float.restype = \
_cpp.im2col_sc_double.restype = \
_cpp.im2col_mc_double.restype = \
_cpp.col2im_sc_float.restype = \
_cpp.col2im_mc_float.restype = \
_cpp.col2im_sc_double.restype = \
_cpp.col2im_mc_double.restype = None

_cpp.im2col_sc_float.argtypes = \
_cpp.im2col_mc_float.argtypes = \
_cpp.col2im_sc_float.argtypes = \
_cpp.col2im_mc_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C')]

_cpp.im2col_sc_double.argtypes = \
_cpp.im2col_mc_double.argtypes = \
_cpp.col2im_sc_double.argtypes = \
_cpp.col2im_mc_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C')]

def im2col_sc(*args):
    """A wrapper of the im2col_sc function."""
    if args[0].dtype == np.float32:
        return _cpp.im2col_sc_float(*args)
    elif args[0].dtype == np.float64:
        return _cpp.im2col_sc_double(*args)
    else:
        raise TypeError('Unsupported type: ' + str(args[0].dtype))

def col2im_sc(*args):
    """A wrapper of the col2im_sc function."""
    if args[0].dtype == np.float32:
        return _cpp.col2im_sc_float(*args)
    elif args[0].dtype == np.float64:
        return _cpp.col2im_sc_double(*args)
    else:
        raise TypeError('Unsupported type: ' + str(args[0].dtype))

def im2col_mc(*args):
    """A wrapper of the im2col_mc function."""
    if args[0].dtype == np.float32:
        return _cpp.im2col_mc_float(*args)
    elif args[0].dtype == np.float64:
        return _cpp.im2col_mc_double(*args)
    else:
        raise TypeError('Unsupported type: ' + str(args[0].dtype))

def col2im_mc(*args):
    """A wrapper of the col2im_mc function."""
    if args[0].dtype == np.float32:
        return _cpp.col2im_mc_float(*args)
    elif args[0].dtype == np.float64:
        return _cpp.col2im_mc_double(*args)
    else:
        raise TypeError('Unsupported type: ' + str(args[0].dtype))

# For convenience, if no mc or sc is specified, we default to mc.
im2col = im2col_mc
col2im = col2im_mc
