"""Implements basic regularizers."""

from decaf import base
import numpy as np


# pylint: disable=R0903
class L2Regularizer(base.Regularizer):
    """The L2 regularization."""
    def reg(self, blob, num_data):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * num_data * 2. * data
        return np.dot(data.flat, data.flat) * self._weight


# pylint: disable=R0903
class L1Regularizer(base.Regularizer):
    """The L1 regularization."""
    def reg(self, blob, num_data):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * num_data * np.sign(data)
        return np.abs(data).sum()
