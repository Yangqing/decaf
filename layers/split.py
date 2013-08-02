"""Implements the split layer."""

from decaf import base

class SplitLayer(base.Layer):
    """A layer that implements the inner product."""

    def __init__(self, **kwargs):
        """Initializes a Split layer.
        """
        base.Layer.__init__(self, **kwargs)
    
    def forward(self, bottom, top):
        """Computes the forward pass.

        The output will simply mirror the input data.
        """
        if len(bottom) != 1:
            raise ValueError(
                'SplitLayer only accepts one input as its bottom.')
        for output in top:
            output.mirror(bottom[0])

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if propagate_down:
            diff = bottom[0].init_diff()
            for single_top in top:
                diff[:] += single_top.diff()
        return 0.

    def update(self):
        """ReLU has nothing to update."""
        pass
