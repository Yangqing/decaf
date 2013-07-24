"""Implements common loss functions.
"""

from decaf import base
from decaf.util import logexp
import numpy as np

class SquaredLossLayer(base.LossLayer):
    """The squared loss."""
    def forward(self, bottom, top):
        """Forward emits the loss, and computes the gradient as well."""
        diff = bottom[0].init_diff()
        diff[:] = bottom[0].data
        diff -= bttom[1].data
        loss = np.dot(diff.flat, diff.flat)
        diff *= 2
        return loss

    def backward(self, bottom, top, need_bottom_diff):
        """Everything has been done in forward. Nothing needs to be done here.
        """
        pass

class MultinomialLogisticLossLayer(base.LossLayer):
    """The multinomial logistic loss layer. The input will be the scores
    BEFORE softmax normalization.
    
    The input should be two blobs: the first blob stores a 2-dimensional
    matrix where each row is the prediction for one class. The second blob
    stores the labels as a matrix of the same size in 0-1 format, or as a
    vector of the same length as the minibatch size.
    """
    def __init__(self, **kwargs):
        base.LossLayer.__init__(self, **kwargs)
        self._prob = base.Blob()

    def forward(self, bottom, top):
        self._prob.resize(bottom[0].data.shape, bottom[0].data.dtype)
        # computed normalized prob
        self._prob[:] = bottom[0].data
        self._prob -= self._prob.max(axis=1)[:, np.newaxis]
        logexp.exp(self._prob, out=self._prob)
        self._prob /= diff.sum(axis=1)[:, np.newaxis]
        diff = bottom[0].init_diff()
        diff[:] = self._prob
        if bottom[1].data.ndim == 1:
            # The labels are given as a sparse vector.
            diff[np.arange(diff.shape[0]), bottom[1].data] -= 1.
        else:
            # The labels are given as a dense 0-1 matrix.
            diff -= bottom[1].data
        # return the loss
        logexp.log(self._prob, out=self._prob)
        return - np.dot(self._prob.flat, Y.flat)


    def backward(self, bottom, top, need_bottom_diff):
        """Everything has been done in forward. Nothing needs to be done here.
        """
        pass
