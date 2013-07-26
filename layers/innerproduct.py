"""Implements the inner product layer."""

from decaf import base
from decaf.util import blasdot
import numpy as np

class InnerProductLayer(base.Layer):
    """A layer that implements the inner product."""

    def __init__(self, **kwargs):
        """Initializes an inner product layer. You need to specify the
        kwarg 'num_output' as the number of output nodes. Optionally, pass
        in a regularizer with keyword 'reg' will add regularization terms
        to the weight (but not bias).
        """
        base.Layer.__init__(self, **kwargs)
        self._num_output = self.spec.get('num_output', 0)
        if self._num_output <= 0:
            raise base.InvalidSpecError(
                'Incorrect or unspecified num_output for %s' % self.name)
        self._reg = self.spec.get('reg', None)
        self._weight = base.Blob()
        self._has_bias = self.spec.get('bias', True)
        if self._has_bias:
            self._bias = base.Blob()
            self._param = [self._weight, self._bias]
        else:
            self._param = [self._weight]
    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        if features.ndim > 2:
            features.shape = (features.shape[0], np.prod(features.shape[1:]))
        output = top[0].init_data(
            (features.shape[0], self._num_output), features.dtype)
        # initialize weights
        if not self._weight.has_data():
            self._weight.init_data(
                (features.shape[1], self._num_output), features.dtype)
        if self._has_bias and not self._bias.has_data():
            self._bias.init_data((self._num_output), features.dtype)
        # computation
        weight = self._weight.data()
        blasdot.dot(features, weight, out=output)
        if self._has_bias:
            output += self._bias.data()

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        # get diff
        top_diff = top[0].diff()
        features = bottom[0].data()
        if features.ndim > 2:
            features.shape = (features.shape[0], np.prod(features.shape[1:]))
        # compute the gradient
        weight_diff = self._weight.init_diff()
        blasdot.dot(features.T, top_diff, out=weight_diff)
        if self._has_bias:
            bias_diff = self._bias.init_diff()
            bias_diff[:] = top_diff.sum(0)
        # If necessary, compute the bottom Blob gradient.
        if propagate_down:
            bottom_diff = bottom[0].init_diff()
            if bottom_diff.shape > 2:
                bottom_diff.shape = (bottom_diff.shape[0],
                                     np.prod(bottom_diff.shape[1:]))
            np.dot(top_diff, weight_diff.T, out=bottom_diff)
        if self._reg is not None:
            return self._reg.reg(self._weight, features.shape[0])
        else:
            return 0.

    def update(self):
        """Updates the parameters."""
        self._weight.update()
        if self._has_bias:
            self._bias.update()

