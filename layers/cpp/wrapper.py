# pylint: disable=C0103
"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os

# first, let's import the library
try:
    _DLL = np.ctypeslib.load_library('libcpputil.so',
            os.path.join(os.path.dirname(__file__)))
except Exception as error:
    raise error


def float_double_strategy(float_func, double_func):
    """Create a function that wraps two functions, one float and one double,
    and decides upon runtime which function to call based on the dtype of the
    first argument.
    """
    def _strategy(*args, **kwargs):
        """The actual strategy."""
        if args[0].dtype == np.float32:
            return float_func(*args, **kwargs)
        elif args[0].dtype == np.float64:
            return double_func(*args, **kwargs)
        else:
            raise TypeError('Unsupported type: ' + str(args[0].dtype))
    return _strategy


################################################################################
# im2col and col2im operation
################################################################################
_DLL.im2col_sc_float.restype = \
_DLL.im2col_mc_float.restype = \
_DLL.im2col_sc_double.restype = \
_DLL.im2col_mc_double.restype = \
_DLL.col2im_sc_float.restype = \
_DLL.col2im_mc_float.restype = \
_DLL.col2im_sc_double.restype = \
_DLL.col2im_mc_double.restype = None

_DLL.im2col_sc_float.argtypes = \
_DLL.im2col_mc_float.argtypes = \
_DLL.col2im_sc_float.argtypes = \
_DLL.col2im_mc_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C')]

_DLL.im2col_sc_double.argtypes = \
_DLL.im2col_mc_double.argtypes = \
_DLL.col2im_sc_double.argtypes = \
_DLL.col2im_mc_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C')]

im2col_sc = float_double_strategy(_DLL.im2col_sc_float,
                                  _DLL.im2col_sc_double)
col2im_sc = float_double_strategy(_DLL.col2im_sc_float,
                                  _DLL.col2im_sc_double)
im2col_mc = float_double_strategy(_DLL.im2col_mc_float,
                                  _DLL.im2col_mc_double)
col2im_mc = float_double_strategy(_DLL.col2im_mc_float,
                                  _DLL.col2im_mc_double)
# For convenience, if no mc or sc is specified, we default to mc.
im2col = im2col_mc
col2im = col2im_mc



################################################################################
# pooling operation
################################################################################
_DLL.maxpooling_forward_float.restype = \
_DLL.maxpooling_backward_float.restype = \
_DLL.avepooling_forward_float.restype = \
_DLL.avepooling_backward_float.restype = \
_DLL.maxpooling_forward_double.restype = \
_DLL.maxpooling_backward_double.restype = \
_DLL.avepooling_forward_double.restype = \
_DLL.avepooling_backward_double.restype = None

_DLL.maxpooling_forward_float.argtypes = \
_DLL.avepooling_forward_float.argtypes = \
_DLL.avepooling_backward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]

_DLL.maxpooling_forward_double.argtypes = \
_DLL.avepooling_forward_double.argtypes = \
_DLL.avepooling_backward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]

_DLL.maxpooling_backward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
_DLL.maxpooling_backward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]

maxpooling_forward = float_double_strategy(_DLL.maxpooling_forward_float,
                                           _DLL.maxpooling_forward_double)
maxpooling_backward = float_double_strategy(_DLL.maxpooling_backward_float,
                                            _DLL.maxpooling_backward_double)
avepooling_forward = float_double_strategy(_DLL.avepooling_forward_float,
                                           _DLL.avepooling_forward_double)
avepooling_backward = float_double_strategy(_DLL.avepooling_backward_float,
                                            _DLL.avepooling_backward_double)

################################################################################
# local contrast normalization operation
################################################################################
_DLL.lrn_forward_float.restype = \
_DLL.lrn_forward_double.restype = \
_DLL.lrn_backward_float.restype = \
_DLL.lrn_backward_double.restype = None

_DLL.lrn_forward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_float]

_DLL.lrn_forward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double]

_DLL.lrn_backward_float.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_float]

_DLL.lrn_backward_double.argtypes = \
    [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
     ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double]

lrn_forward = float_double_strategy(_DLL.lrn_forward_float,
                                    _DLL.lrn_forward_double)
lrn_backward = float_double_strategy(_DLL.lrn_backward_float,
                                     _DLL.lrn_backward_double)
