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
_DLL.maxpooling_forward.restype = \
_DLL.maxpooling_backward.restype = \
_DLL.avepooling_forward.restype = \
_DLL.avepooling_backward.restype = None

def maxpooling_forward(image, pooled, psize, stride):
    height, width, channels = image.shape
    _DLL.maxpooling_forward(ct.c_int(image.itemsize),
                            image.ctypes.data_as(ct.c_void_p),
                            pooled.ctypes.data_as(ct.c_void_p),
                            ct.c_int(height),
                            ct.c_int(width),
                            ct.c_int(channels),
                            ct.c_int(psize),
                            ct.c_int(stride))

def avepooling_forward(image, pooled, psize, stride):
    height, width, channels = image.shape
    _DLL.avepooling_forward(ct.c_int(image.itemsize),
                            image.ctypes.data_as(ct.c_void_p),
                            pooled.ctypes.data_as(ct.c_void_p),
                            ct.c_int(height),
                            ct.c_int(width),
                            ct.c_int(channels),
                            ct.c_int(psize),
                            ct.c_int(stride))

def maxpooling_backward(image, pooled, image_diff, pooled_diff, psize,
                        stride):
    height, width, channels = image.shape
    _DLL.maxpooling_backward(ct.c_int(image.itemsize),
                             image.ctypes.data_as(ct.c_void_p),
                             pooled.ctypes.data_as(ct.c_void_p),
                             image_diff.ctypes.data_as(ct.c_void_p),
                             pooled_diff.ctypes.data_as(ct.c_void_p),
                             ct.c_int(height),
                             ct.c_int(width),
                             ct.c_int(channels),
                             ct.c_int(psize),
                             ct.c_int(stride))

def avepooling_backward(image_diff, pooled_diff, psize, stride):
    height, width, channels = image_diff.shape
    _DLL.avepooling_backward(ct.c_int(image_diff.itemsize),
                             image_diff.ctypes.data_as(ct.c_void_p),
                             pooled_diff.ctypes.data_as(ct.c_void_p),
                             ct.c_int(height),
                             ct.c_int(width),
                             ct.c_int(channels),
                             ct.c_int(psize),
                             ct.c_int(stride))



################################################################################
# local contrast normalization operation
################################################################################
_DLL.lrn_forward.restype = \
_DLL.lrn_backward.restype = None

def lrn_forward(bottom, top, scale, size, alpha, beta):
    _DLL.lrn_forward(ct.c_int(bottom.itemsize),
                     bottom.ctypes.data_as(ct.c_void_p),
                     top.ctypes.data_as(ct.c_void_p),
                     scale.ctypes.data_as(ct.c_void_p),
                     ct.c_int(bottom.size / bottom.shape[-1]),
                     ct.c_int(bottom.shape[-1]),
                     ct.c_int(size),
                     ct.c_double(alpha),
                     ct.c_double(beta))


def lrn_backward(bottom, top, bottom_diff, top_diff, scale, size, alpha, beta):
    _DLL.lrn_backward(ct.c_int(bottom.itemsize),
                     bottom.ctypes.data_as(ct.c_void_p),
                     top.ctypes.data_as(ct.c_void_p),
                     bottom_diff.ctypes.data_as(ct.c_void_p),
                     top_diff.ctypes.data_as(ct.c_void_p),
                     scale.ctypes.data_as(ct.c_void_p),
                     ct.c_int(bottom.size / bottom.shape[-1]),
                     ct.c_int(bottom.shape[-1]),
                     ct.c_int(size),
                     ct.c_double(alpha),
                     ct.c_double(beta))
