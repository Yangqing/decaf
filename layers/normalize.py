"""Implements the Mean and variance normalization layer."""

from decaf import base
import numpy as np
import numexpr

class MeanNormalizeLayer(base.Layer):
    """ A Layer that removes the mean along the mast dimension.
    """
    def __init__(self, **kwargs):
        base.Layer.__init__(self, **kwargs)

    def forward(self, bottom, top):
        """Computes the backward pass."""
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype)
        # Create 2-dimenisonal views of the features and outputs.
        features.shape = (np.prod(features.shape[:-1]), features.shape[-1])
        output.shape = features.shape
        output[:] = features
        output -= features.mean(axis=1)[:, np.newaxis]

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if propagate_down:
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff()
            top_diff.shape = (np.prod(top_diff.shape[:-1]), top_diff.shape[-1])
            bottom_diff.shape = top_diff.shape
            bottom_diff[:] = top_diff
            bottom_diff -= (top_diff.sum(1) / top_diff.shape[-1])[:, np.newaxis]
        return 0.
        
    def update(self):
        """Has nothing to update."""
        pass



class ResponseNormalizeLayer(base.Layer):
    """A layer that normalizes the last dimension. For a vector x, it is normalized as
    x / (smooth + std(x)).
    """
    def __init__(self, **kwargs):
        """Initalizes the layer. 
        
        kwargs:
        'smooth': the smoothness term added to the norm.
        """
        base.Layer.__init__(self, **kwargs)
        self._std = None
        self._stdsmooth = None

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype)
        # Create 2-dimenisonal views of the features and outputs.
        features.shape = (np.prod(features.shape[:-1]), features.shape[-1])
        output.shape = features.shape
        self._stdsmooth = features.std(axis=1) + \
                self.spec.get('smooth', np.finfo(self._std.dtype).eps)
        output[:] = features
        output /= self._stdsmooth[:, np.newaxis]
    
    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        raise NotImplementedError("Not Implemented yet.")

    def update(self):
        """Has nothing to update."""
        pass
