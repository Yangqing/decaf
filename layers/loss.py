"""Implements common loss functions.
"""

from decaf import base
from decaf.util import logexp
import numpy as np

class SquaredLossLayer(base.LossLayer):
    """The squared loss."""
    def backward(self, bottom, top, need_bottom_diff):
        if not need_bottom_diff:
            return
        bottom[0].init_diff()
        bottom[0].diff[:] = bottom[0].data
        bottom[0].diff -= bottom[1].data
        bottom[0].diff *= 2.

class MultinomialLogisticLossLayer(base.LossLayer):
    """The multinomial logistic loss layer. The input will be the scores
    BEFORE softmax normalization.
    
    The input should be two blobs: the first blob stores a 2-dimensional
    matrix where each row is the prediction for one class. The second blob
    stores the labels as a matrix of the same size in 0-1 format, or as a
    vector of the same length as the minibatch size.
    """
    def backward(self, bottom, top, need_bottom_diff):
        if not need_bottom_diff:
            return
        diff = bottom[0].init_diff()
        diff[:] = bottom[0].data
        diff -= diff.max(axis=1)[:, np.newaxis]
        logexp.exp(diff, out=diff)
        diff /= diff.sum(axis=1)[:, np.newaxis]
        if bottom[1].data.ndim == 1:
            # The labels are given as a sparse vector.
            diff[np.arange(diff.shape[0]), bottom[1].data] -= 1.
        else:
            # The labels are given as a dense 0-1 matrix.
            diff -= bottom[1].data
