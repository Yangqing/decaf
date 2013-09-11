import cuda
import numpy as np

a = np.random.rand(4,3)
b = np.random.rand(3,4)
out = np.empty((4,4), order='F')
cuda.dot(a,b,out=out)
