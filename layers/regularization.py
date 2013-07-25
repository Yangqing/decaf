"""Implements basic regularizers."""

from decaf import base
import numpy as np


class L2Regularizer(base.Regularizer):
    def reg(self, blob):
        blob.diff += self._weight * 2. * blob.data
        return np.dot(blob.data.flat, blob.data.flat) * self._weight

class L1Regularizer(base.Regularizer):
    def reg(self, blob):
        blob.diff += self._weight * np.sign(blob.data)
        return np.abs(blob.data)
        gradient = np.sign(blob.data).sum() * self._weight
