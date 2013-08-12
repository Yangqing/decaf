"""Implements the pooling layer."""

from decaf import base
from decaf.layers.cpp import wrapper

class PoolingLayer(base.Layer):
    """A layer that implements the pooling function."""

    def __init__(self, **kwargs):
        """Initializes the pooling layer.

        kwargs:
            name: the name of the layer.
            psize: the pooling size. Pooling regions will be square shaped.
            stride: the pooling stride. If not given, it will be the same as
                the psize.
            mode: 'max' or 'full'.
        """
        base.Layer.__init__(self, **kwargs)
        self._psize = self.spec['psize']
        self._stride = self.spec.get('stride', self._psize)
        self._mode = self.spec['mode']
        if self._stride > self._psize:
            raise ValueError(
                    'Currently, we do not support stride > psize case.')
        if self._psize <= 1:
            raise ValueError('Invalid pool size. Pool size should > 1.')
        if self._stride <= 0:
            raise ValueError('Invalid stride size. Stride size should > 0.')
    
    def forward(self, bottom, top):
        """Runs the forward pass."""
        bottom_data = bottom[0].data()
        num, height, width, nchannels = bottom_data.shape
        pooled_height = (height - self._psize) / self._stride + 1
        pooled_width = (width - self._psize) / self._stride + 1
        top_data = top[0].init_data(
            (num, pooled_height, pooled_width, nchannels),
            dtype=bottom_data.dtype)
        if self._mode == 'max':
            for i in range(num):
                wrapper.maxpooling_forward(bottom_data[i], top_data[i],
                                           height, width, nchannels,
                                           self._psize, self._stride)
        elif self._mode == 'ave':
            for i in range(num):
                wrapper.avepooling_forward(bottom_data[i], top_data[i],
                                           height, width, nchannels,
                                           self._psize, self._stride)
        else:
            raise ValueError('Unknown mode: %s.' % self._mode)
        return

    def backward(self, bottom, top, propagate_down):
        """Runs the backward pass."""
        if propagate_down:
            bottom_diff = bottom[0].init_diff()
            top_diff = top[0].diff()
            num, height, width, nchannels = bottom_diff.shape
            if self._mode == 'max':
                for i in range(num):
                    bottom_data = bottom[0].data()
                    top_data = top[0].data()
                    wrapper.maxpooling_backward(
                            bottom_data[i], top_data[i], bottom_diff[i],
                            top_diff[i], height, width, nchannels,
                            self._psize, self._stride)
            elif self._mode == 'ave':
                for i in range(num):
                    wrapper.avepooling_backward(
                            bottom_diff[i], top_diff[i], height, width,
                            nchannels, self._psize, self._stride)
            else:
                raise ValueError('Unknown mode: %s.' % self._mode)
        return 0.

    def update(self):
        pass
