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

def float_double_strategy(float_func, double_func):
    def _strategy(*args, **kwargs):
        if args[0].dtype == np.float32:
            return float_func(*args, **kwargs)
        elif args[0].dtype == np.float64:
            return double_func(*args, **kwargs)
        else:
            raise TypeError('Unsupported type: ' + str(dtype))
    return _strategy


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

im2col_sc = float_double_strategy(_cpp.im2col_sc_float,
                                  _cpp.im2col_sc_double)
col2im_sc = float_double_strategy(_cpp.col2im_sc_float,
                                  _cpp.col2im_sc_double)
im2col_mc = float_double_strategy(_cpp.im2col_mc_float,
                                  _cpp.im2col_mc_double)
col2im_mc = float_double_strategy(_cpp.col2im_mc_float,
                                  _cpp.col2im_mc_double)
# For convenience, if no mc or sc is specified, we default to mc.
im2col = im2col_mc
col2im = col2im_mc



################################################################################
# pooling operation
################################################################################
_cpp.maxpooling_forward_float.restype = \
_cpp.maxpooling_backward_float.restype = \
_cpp.avepooling_forward_float.restype = \
_cpp.avepooling_backward_float.restype = \
_cpp.maxpooling_forward_double.restype = \
_cpp.maxpooling_backward_double.restype = \
_cpp.avepooling_forward_double.restype = \
_cpp.avepooling_backward_double.restype = None

_cpp.maxpooling_forward_float.argtypes = \
_cpp.avepooling_forward_float.argtypes = \
_cpp.avepooling_backward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]

_cpp.maxpooling_forward_double.argtypes = \
_cpp.avepooling_forward_double.argtypes = \
_cpp.avepooling_backward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]

_cpp.maxpooling_backward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
_cpp.maxpooling_backward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]

maxpooling_forward = float_double_strategy(_cpp.maxpooling_forward_float,
                                           _cpp.maxpooling_forward_double)
maxpooling_backward = float_double_strategy(_cpp.maxpooling_backward_float,
                                            _cpp.maxpooling_backward_double)
avepooling_forward = float_double_strategy(_cpp.avepooling_forward_float,
                                           _cpp.avepooling_forward_double)
avepooling_backward = float_double_strategy(_cpp.avepooling_backward_float,
                                            _cpp.avepooling_backward_double)

################################################################################
# local contrast normalization operation
################################################################################
_cpp.lrn_forward_float.restype = \
_cpp.lrn_forward_double.restype = \
_cpp.lrn_backward_float.restype = \
_cpp.lrn_backward_double.restype = None

_cpp.lrn_forward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_float]

_cpp.lrn_forward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double]

_cpp.lrn_backward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_float]

_cpp.lrn_backward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double]

lrn_forward = float_double_strategy(_cpp.lrn_forward_float, _cpp.lrn_forward_double)
lrn_backward = float_double_strategy(_cpp.lrn_backward_float, _cpp.lrn_backward_double)
