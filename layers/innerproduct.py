"""Implements the inner product layer."""

from decaf import base
from decaf.util import blasdot

class InnerProductLayer(base.Layer):
    def __init__(self, **kwargs):
        """Initializes an inner product layer. You need to specify the
        kwarg 'num_output' as the number of output nodes.
        """
        Layer.__init__(self, **kwargs)
        self._num_output = self.spec.get('num_output', 0)
        if self._num_output <= 0:
            raise InvalidSpecError(
                'Incorrect or unspecified num_output for %s' % self.name)
        self._w = base.Blob()
        self._has_bias = self.spec.get('bias', True)
        if self._has_bias:
            self._b = base.Blob()
            self._param = [self._w, self._b]
        else:
            self._param = [self._w]
    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        bottom_data = bottom[0].data.view()
        bottom_data.shape = (bottom_data.shape[0], bottom_data.shape[1:])
        top[0].resize((bottom_data.shape[0], self._num_output),
                      bottom_data.dtype)
        top_data = top[0].data
        self._w.resize((bottom_data.shape[1], self._num_output),
                       bottom_data.dtype)
        np.dot(bottom_data, self._w.data, out=top_data)
        if self._has_bias:
            self._b.resize((self._num_output), bottom_data.dtype)
            top_data += self._b


    def backward(self, bottom, top, need_bottom_diff):
        """Computes the backward pass."""
        top_diff = top[0].diff.view()
        bottom_data = bottom[0].data.view()
        bottom_data.shape = (bottom_data.shape[0], bottom_data.shape[1:])
        # compute the gradient
        self._w.init_diff()
        blasdot.dot(bottom_data.T, top_diff, out=self._w.diff)
        if self._has_bias:
            self._b.init_diff()
            self._b.diff[:] = top_diff.sum(0)
        # If necessary, compute the bottom Blob gradient.
        if need_bottom_diff:
            bottom[0].init_diff()
            bottom_diff = bottom[0].diff.view()
            bottom_diff.shape = (bottom_diff.shape[0], bottom_diff.shape[1:])
            blasdot.dot(top_diff, self._w.diff.T, out=bottom_diff)

    def update(self):
        """Updates the parameters."""
        self._w.data += self._w.diff
        if self._has_bias:
            self._b.data += self._b.diff

