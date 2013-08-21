# pylint: disable=C0103
"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
from decaf import base
import numpy as np
import os

# first, let's import the library
try:
    _DLL = np.ctypeslib.load_library('libdecafcuda.so',
            os.path.join(os.path.dirname(__file__)))
except Exception as error:
    raise error

class DecafCudaError(base.DecafError):
    """An error that will be raised if anything at cuda is wrong."""
    pass

################################################################################
# init_cuda
################################################################################
_DLL.init_cuda.restype = ct.c_int

def init_cuda():
    if _DLL.init_cuda():
        raise DecafCudaError('Initialization Failed.')
