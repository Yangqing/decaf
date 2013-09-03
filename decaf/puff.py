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
    def __init__(self, name, start = None, end = None):
        """Initializes the puff object.

        Input:
            name: the puff filename to be read.
            start: (optional) the local range start.
            end: (optional) the local range end.
        """
        if name.endswith('.puff'):
            name = name[:-5]
        # shape is the shape of a single data point.
        self._shape = None
        # step is an internal variable that indicates how many bytes we need
        # to jump over per data point
        self._step = None
        # num_data is the total number of data in the file
        self._num_data = None
        # the following variables are used to slice a puff
        self._start = None
        self._end = None
        # the current index of the data.
        self._curr = None
        # the number of local data
        self._num_local_data = None
        # dtype is the data type of the data
        self._dtype = None
        # the fid for the opened file
        self._fid = None
        self.open(name)
        self.set_range(start, end)

    def set_range(self, start, end):
        """sets the range that we will read data from."""
        if start is not None:
            if start > self._num_data:
                raise ValueError('Invalid start index.')
            else:
                self._start = start
                self.seek(self._start)
                self._curr = self._start
        if end is not None:
            if end > start and end <= self._num_data:
                self._end = end
            else:
                raise ValueError('Invalid end index.')
        self._num_local_data = self._end - self._start

    def num_data(self):
        """Return the number of data."""
        return self._num_data
    
    def shape(self):
        """Return the shape of a single data point."""
        return self._shape

    def dtype(self):
        """Return the dtype of the data."""
        return self._dtype

    def num_local_data(self):
        """Returns the number of local data."""
        return self._num_local_data

    def open(self, name):
        """Opens a puff data: it is composed of two files, name.puff and
        name.icing. The open function will set the range to all the data
        points - use set_range() to specify a custom range to read from.
        """
        icing = pickle.load(open(name + '.icing'))
        self._shape = icing['shape']
        self._dtype = icing['dtype']
        self._num_data = icing['num']
        self._step = reduce(mul, self._shape, 1)
        self._fid = open(name + '.puff', 'rb')
        self._start = 0
        self._end = self._num_data
        self._num_local_data = self._num_data
        self._curr = 0

    def seek(self, offset):
        """Seek to the beginning of the offset-th data point."""
        if offset < self._start or offset >= self._end:
            raise ValueError('Offset should lie in the data range.')
        self._fid.seek(offset * self._step * self._dtype.itemsize)
        self._curr = offset

    def read(self, count):
        """Read the specified number of data and return as a numpy array."""
        if count > self._num_local_data:
            raise ValueError('Not enough data points to read: count %d, limit'
                             ' %d.' % (count, self._num_local_data))
        if self._curr + count <= self._end:
            data = np.fromfile(self._fid, self._dtype, count * self._step)
            self._curr += count
            if self._curr == self._end:
                # When everything is read, we restart from the head.
                self.seek(self._start)
        else:
            part = self._end - self._curr
            data = np.vstack((self.read(part),
                              self.read(count - part)))
        return data.reshape((count,) + self._shape)

    def read_all(self):
        """Reads all the data from the file."""
        self.seek(self._start)
        return self.read(self._num_local_data)


class PuffStreamedWriter(object):
    """A streamed writer to write a large puff incrementally."""
    def __init__(self, name):
        self._shape = None
        self._num_data = 0
        self._dtype = None
        self._fid = open(name + '.puff', 'wb')
        self._name = name
    
    def check_validity(self, arr):
        """Checks if the data is valid."""
        if self._shape is None:
            self._shape = arr.shape
            self._dtype = arr.dtype
        else:
            if self._shape != arr.shape or self._dtype != arr.dtype:
                raise TypeError('Array invalid with previous inputs! '
                                'Previous: %s, %s, current: %s %s' %
                                (str(self._shape), str(self._dtype),
                                 str(arr.shape), str(arr.dtype)))

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
        with open(self._name + '.icing', 'w') as fid:
            pickle.dump({'shape': self._shape,
                         'dtype': self._dtype,
                         'num': self._num_data}, fid)

def write_puff(arr, name):
    """Write a single numpy array to puff format."""
    writer = PuffStreamedWriter(name)
    writer.write_batch(arr)
    writer.finish()
