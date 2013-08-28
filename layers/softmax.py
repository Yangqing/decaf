"""Implements the softmax function.
"""

from decaf import base
from decaf.util import logexp
import numpy as np

class SoftmaxLayer(base.Layer):
    """A layer that implements the softmax function."""

    def __init__(self, **kwargs):
        """Initializes a softmax layer.

        kwargs:
            name: the layer name.
        """
        base.Layer.__init__(self, **kwargs)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        pred = bottom[0].data()
        prob = top[0].init_data(features.shape, features.dtype, setdata=False)
        prob[:] = pred
        # normalize by subtracting the max to suppress numerical issues
        prob -= prob.max(axis=1)[:, np.newaxis]
        logexp.exp(prob, out=prob)
        prob /= prob.sum(axis=1)[:, np.newaxis]

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        bottom_diff = bottom[0].init_diff(setzero=False)
        raise NotImplementedError 
        return 0.

    def update(self):
        """Softmax has nothing to update."""
        pass
