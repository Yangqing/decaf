"""The module that implements Blob, the basic component that contains a piece
of matrix in addition to its gradients.
"""

import cPickle as pickle
import numpy as np

# pylint: disable=R0903
class Blob(object):
    """Blob is the data structure that holds a piece of numpy array as well as
    its gradient so that we can accumulate and pass around data more easily.

    We define two numpy matrices: one is data, which stores the data in the
    current blob; the other is diff (short for difference): when a network
    runs its forward and backward pass, diff will store the gradient value;
    when a solver goes through the blobs, diff will then be replaced with the
    value to update.

    The diff matrix will not be created unless you explicitly run init_diff,
    as many Blobs do not need the gradients to be computed.
    """
    def __init__(self, shape=None, dtype=np.float64, filler=None):
        self._data = None
        self._diff = None
        self._filler = filler
        if shape is not None:
            self.init_data(shape, dtype)

    @staticmethod
    def blob_like(source_blob):
        return Blob(source_blob._data.shape, source_blob._data.dtype,
                    source_blob._filler)

    def clear(self):
        """Clears a blob data."""
        self._data = None
        self._diff = None

    def mirror(self, input_array, shape=None):
        """Create the data as a view of the input array. This is useful to
        save space and avoid duplication for data layers.
        """
        if isinstance(input_array, Blob):
            self._data = input_array.data()
        else:
            self._data = input_array.view()
        if shape is not None:
            self._data.shape = shape
        return self.data()
   
    def mirror_diff(self, input_array, shape=None):
        """Create the diff as a view of the input array's diff. This is useful
        to save space and avoid duplication for data layers.
        """
        if isinstance(input_array, Blob):
            self._diff = input_array.diff()
        else:
            self._diff = input_array.view()
        if shape is not None:
            self._diff.shape = shape
        return self.diff()

    def has_data(self):
        """Checks if the blob has data."""
        return self._data is not None
    
    def data(self):
        """Returns a view of the data."""
        return self._data.view()

    def has_diff(self):
        """Checks if the blob has diff."""
        return self._diff is not None

    def diff(self):
        """Returns the diff."""
        return self._diff.view()

    def update(self):
        """Update the data field by SUBTRACTING diff to it.
        
        Note that diff is often used to store the gradients, and most often
        we will perform MINIMIZATION. This is why we always do subtraction
        here.
        """
        self._data -= self._diff

    def init_data(self, shape, dtype, setdata=True):
        """Initializes the data if necessary. The filler will be always
        called even if no reallocation of data takes place.
        """
        if not(self.has_data() and self._data.shape == shape and \
           self._data.dtype == dtype):
            self._data = np.empty(shape, dtype)
        if setdata:
            if self._filler is not None:
                self._filler.fill(self._data)
            else:
                self._data[:] = 0
        return self.data()

    def init_diff(self, setzero=True):
        """Initialize the diff in the same format as data.
        
        Returns diff for easy access.
        """
        if not self.has_data():
            raise ValueError('The data should be initialized first!')
        if self.has_diff() and self._diff.shape == self._data.shape and \
           self._diff.dtype == self._data.dtype:
            if setzero:
                self._diff[:] = 0
        else:
            self._diff = np.zeros(self._data.shape, self._data.dtype)
        return self.diff()

    def swap_data(self, other_blob):
        """swaps the data between two blobs."""
        if not(self.has_data() and other_blob.has_data() and
               self._data.dtype == other_blob._data.dtype and
               self._data.shape == other_blob._data.shape):
            raise DecafError('Attempting to swap incompatible blobs.')
        self._data, other_blob._data = other_blob._data, self._data
    
    def __getstate__(self):
        """When pickling, we will not store the diff field."""
        dictionary = dict(self.__dict__)
        dictionary['_diff'] = None
        return dictionary


