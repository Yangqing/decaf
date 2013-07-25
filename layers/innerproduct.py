"""Implements the inner product layer."""

from decaf import base
from decaf.util import blasdot

class InnerProductLayer(base.Layer):
    """A layer that implements the inner product."""

    def __init__(self, **kwargs):
        """Initializes an inner product layer. You need to specify the
        kwarg 'num_output' as the number of output nodes.
        """
        base.Layer.__init__(self, **kwargs)
        self._num_output = self.spec.get('num_output', 0)
        if self._num_output <= 0:
            raise base.InvalidSpecError(
                'Incorrect or unspecified num_output for %s' % self.name)
        self._weight = base.Blob()
        self._has_bias = self.spec.get('bias', True)
        if self._has_bias:
            self._bias = base.Blob()
            self._param = [self._weight, self._bias]
        else:
            self._param = [self._weight]
    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        bottom_data = bottom[0].data.view()
        bottom_data.shape = (bottom_data.shape[0], bottom_data.shape[1:])
        top[0].resize((bottom_data.shape[0], self._num_output),
                      bottom_data.dtype)
        top_data = top[0].data
        self._weight.resize((bottom_data.shape[1], self._num_output),
                       bottom_data.dtype)
        blasdot.dot(bottom_data, self._weight.data, out=top_data)
        if self._has_bias:
            self._bias.resize((self._num_output), bottom_data.dtype)
            top_data += self._bias


    def backward(self, bottom, top, need_bottom_diff):
        """Computes the backward pass."""
        top_diff = top[0].diff.view()
        bottom_data = bottom[0].data.view()
        bottom_data.shape = (bottom_data.shape[0], bottom_data.shape[1:])
        # compute the gradient
        self._weight.init_diff()
        blasdot.dot(bottom_data.T, top_diff, out=self._weight.diff)
        if self._has_bias:
            self._bias.init_diff()
            self._bias.diff[:] = top_diff.sum(0)
        # If necessary, compute the bottom Blob gradient.
        if need_bottom_diff:
            bottom[0].init_diff()
            bottom_diff = bottom[0].diff.view()
            bottom_diff.shape = (bottom_diff.shape[0], bottom_diff.shape[1:])
            blasdot.dot(top_diff, self._weight.diff.T, out=bottom_diff)

    def update(self):
        """Updates the parameters."""
        self._weight.data += self._weight.diff
        if self._has_bias:
            self._bias.data += self._bias.diff

