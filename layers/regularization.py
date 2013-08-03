"""Implements basic regularizers."""

from decaf import base
import numpy as np
from decaf.util import logexp


# pylint: disable=R0903
class L2Regularizer(base.Regularizer):
    """The L2 regularization."""
    def reg(self, blob):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * 2. * data
        return np.dot(data.flat, data.flat) * self._weight


# pylint: disable=R0903
class L1Regularizer(base.Regularizer):
    """The L1 regularization."""
    def reg(self, blob):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * np.sign(data)
        return np.abs(data).sum() * self._weight


class AutoencoderRegularizer(base.Regularizer):
    """The sparse autoencoder regularization term."""
    def reg(self, blob):
        """The reg function."""
        data = blob.data()
        diff = blob.diff()
        data_mean = data.mean(axis=0)
        # we clip it to avoid overflow
        np.clip(data_mean, np.finfo(data_mean.dtype).eps,
                1. - np.finfo(data_mean.dtype).eps,
                out=data_mean)
        neg_data_mean = 1. - data_mean
        ratio = self.spec['ratio']
        log_mean = logexp.log(data_mean)
        log_neg_mean = logexp.log(neg_data_mean)
        loss = (ratio - 1.) * log_neg_mean.sum() - ratio * log_mean.sum() \
               + ((ratio * np.log(ratio) + (1. - ratio) * np.log(1. - ratio))
                  * data_mean.size)
        loss *= data.shape[0]
        data_diff = (1.-ratio) / neg_data_mean - ratio / data_mean
        data_diff *= self._weight
        diff += data_diff
        return loss * self._weight

