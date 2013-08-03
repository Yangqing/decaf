"""Implements common loss functions.
"""

from decaf import base
from decaf.util import logexp
import numpy as np

class SquaredLossLayer(base.LossLayer):
    """The squared loss. Following conventions, we actually compute
    the one-half of the squared loss.
    """
    def forward(self, bottom, top):
        """Forward emits the loss, and computes the gradient as well."""
        diff = bottom[0].init_diff()
        diff[:] = bottom[0].data()
        diff -= bottom[1].data()
        self._loss = np.dot(diff.flat, diff.flat) / 2. / diff.shape[0]
        diff *= 1. / diff.shape[0]



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

    def __getstate__(self, **kwargs):
        self._prob.clear()
        return self.__dict__

    def forward(self, bottom, top):
        pred = bottom[0].data()
        prob = self._prob.init_data(
            pred.shape, pred.dtype)
        prob[:] = pred
        prob -= prob.max(axis=1)[:, np.newaxis]
        logexp.exp(prob, out=prob)
        prob /= prob.sum(axis=1)[:, np.newaxis]
        diff = bottom[0].init_diff()
        diff[:] = prob
        logexp.log(prob, out=prob)
        label = bottom[1].data()
        if label.ndim == 1:
            # The labels are given as a sparse vector.
            diff[np.arange(diff.shape[0]), label] -= 1.
            self._loss = -prob[np.arange(diff.shape[0]), label].sum()
        else:
            # The labels are given as a dense matrix.
            diff -= label
            self._loss = -np.dot(prob.flat, label.flat)
        # finally, scale down by the number of data points
        diff *= 1. / diff.shape[0]
        self._loss /= diff.shape[0]
