"""Puff defines a purely unformatted file format accompanying decaf for easier
and faster access of numpy arrays.
"""
import cPickle as pickle
import numpy as np
from operator import mul

class Puff(object):
    """The puff class. It defines a simple interface that stores numpy arrays in
    its raw form.
    """
    def __init__(self, name):
        # shape is the shape of a single data point.
        self._shape = None
        # step is an internal variable that indicates how many bytes we need
        # to jump over per data point
        self._step = None
        # num_data is the total number of data in the file
        self._num_data = None
        # dtype is the data type of the data
        self._dtype = None
        # the fid for the opened file
        self._fid = None
        # the current index of the data.
        self._curr = None
        self.open(name)

    def open(self, name):
        """Opens a puff data: it is composed of two files, name and name.meta.
        """
        meta = pickle.load(open(name + '.meta'))
        self._shape = meta['shape']
        self._dtype = meta['dtype']
        self._num_data = meta['num']
        self._step = reduce(mul, self._shape, 1)
        self._fid = open(name, 'rb')
        self._curr = 0

    def seek(self, offset):
        """Seek to the beginning of the offset-th data point."""
        self._fid.seek(offset * self._step * self._dtype.itemsize)
        self._curr = offset

    def read(self, count):
        """Read the specified number of data and return as a numpy array."""
        if count > self._num_data:
            raise ValueError('Not enough data points to read.')
        if self._curr + count <= self._num_data:
            data = np.fromfile(self._fid, self._dtype, count * self._step)
            self._curr += count
            if self._curr == self._num_data:
                # When everything is read, we restart from the head.
                self.seek(0)
        else:
            part = self._num_data - self._curr
            data = np.vstack((self.read(part),
                              self.read(count - part)))
        return data.reshape((count,) + self._shape)

    def read_all(self):
        """Reads all the data from the file."""
        self.seek(0)
        return self.read(self._num_data)


class PuffStreamedWriter(object):
    """A streamed writer to write a large puff incrementally."""
    def __init__(self, name):
        self._shape = None
        self._num_data = 0
        self._dtype = None
        self._fid = open(name, 'wb')
        self._name = name
    
    def check_validity(self, arr):
        """Checks if the data is valid."""
        if self._shape is None:
            self._shape = arr.shape
            self._dtype = arr.dtype
        else:
            if self._shape != arr.shape or self._dtype != arr.dtype:
                raise TypeError('Array invalid with previous inputs!')

    def write_single(self, arr):
        """Write a single data point."""
        self.check_validity(arr)
        arr.tofile(self._fid)
        self._num_data += 1

    def write_batch(self, arr):
        """Write a bunch of data points to file."""
        self.check_validity(arr[0])
        arr.tofile(self._fid)
        self._num_data += arr.shape[0]

    def finish(self):
        """Finishes a Puff write."""
        if self._num_data == 0:
            raise ValueError('Nothing is written!')
        self._fid.close()
        with open(self._name + '.meta', 'w') as fid:
            pickle.dump({'shape': self._shape,
                         'dtype': self._dtype,
                         'num': self._num_data}, fid)

def write_puff(arr, name):
    """Write a single numpy array to puff format."""
    writer = PuffStreamedWriter(name)
    writer.write_batch(arr)
    writer.finish()
