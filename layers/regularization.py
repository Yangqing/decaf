"""Implements basic regularizers."""

from decaf import base
import numpy as np


class L2Regularizer(base.Regularizer):
    def reg(self, blob, num_data):
        data = blob.data()
        diff = blob.diff()
        diff += self._weight * num_data * 2. * data
        return np.dot(data.flat, data.flat) * self._weight

class L1Regularizer(base.Regularizer):
    def reg(self, blob, num_data):
        data = blob.data()
        diff = blob.diff()
        diff += self._weight * num_data * np.sign(data)
        return np.abs(data)
        gradient = np.sign(data).sum() * self._weight
