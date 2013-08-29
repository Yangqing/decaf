"""Implements the dropout layer."""

from decaf import base
from decaf.layers import fillers
import numpy as np
import numexpr

class DropoutLayer(base.Layer):
    """A layer that implements the dropout."""

    def __init__(self, **kwargs):
        """Initializes a Dropout layer.

        kwargs:
            name: the layer name.
            ratio: the ratio to carry out dropout.
            debug_freeze: a debug flag. If set True, the mask will only
                be generated once when running. You should not use it other
                than purposes like gradient check.
        """
        base.Layer.__init__(self, **kwargs)
        filler = fillers.DropoutFiller(ratio=self.spec['ratio'])
        self._mask = base.Blob(filler=filler)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and output
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype, setdata=False)
        if not (self.spec.get('debug_freeze', False) and self._mask.has_data()):
            mask = self._mask.init_data(features.shape, np.bool)
        else:
            mask = self._mask.data()
        numexpr.evaluate('features * mask', out=output)

    def predict(self, bottom, top):
        """The dropout predict pass. It will not randomly shut off features,
        but will instead scale data according to the ratio.
        """
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype, setdata=False)
        ratio = self.spec['ratio']
        numexpr.evaluate('features * ratio', out=output)

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        bottom_diff = bottom[0].init_diff(setzero=False)
        mask = self._mask.data()
        numexpr.evaluate('top_diff * mask', out=bottom_diff) 
        return 0.

    def update(self):
        """Dropout has nothing to update."""
        pass
