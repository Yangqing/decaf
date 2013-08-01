import os
from skimage import io
import numpy as np

_data_path = os.path.join(os.path.dirname(__file__), '_data')

def lena():
    """A simple function to return the lena image."""
    filename = os.path.join(_data_path, 'lena.png')
    return io.imread(filename)

def whitened_images():
    """Returns the whitened images provided in the Sparsenet website:
        http://redwood.berkeley.edu/bruno/sparsenet/
    The returned data will be in the shape (10,512,512,1) to fit
    the blob convension.
    """
    npzdata = np.load(os.path.join(_data_path, 'whitened_images.npz'))
    return npzdata['images']
