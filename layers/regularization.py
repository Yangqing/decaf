"""Implements basic regularizers."""

from decaf import base
import numpy as np
from decaf.util import logexp

def make_regularize_layer(regularizer_class):
    def _make_layer(**kwargs):
        return base.RegularizeLayer(name=kwargs['name'],
                                    reg=regularizer_class(**kwargs))
    return _make_layer


# pylint: disable=R0903
class L2Regularizer(base.Regularizer):
    """The L2 regularization."""
    def reg(self, blob, scale):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * scale * 2. * data
        return np.dot(data.flat, data.flat) * self._weight * scale

L2RegularizeLayer = make_regularize_layer(L2Regularizer)

# pylint: disable=R0903
class L1Regularizer(base.Regularizer):
    """The L1 regularization."""
    def reg(self, blob, scale):
        """The reg function."""
        data = blob.data()
        #pylint: disable=W0612
        diff = blob.diff()
        diff += self._weight * scale * np.sign(data)
        return np.abs(data).sum() * self._weight * scale

L1RegularizeLayer = make_regularize_layer(L1Regularizer)

class AutoencoderRegularizer(base.Regularizer):
    """The sparse autoencoder regularization term."""
    def reg(self, blob, scale):
        """The reg function."""
        data = blob.data()
        diff = blob.diff()
        data_mean = data.mean(axis=0)
        neg_data_mean = 1. - data_mean
        ratio = self.spec['ratio']
        # log(data_mean / ratio) and log((1-data_mean) / (1-ratio))
        log_divide_1 = logexp.log(data_mean / ratio)
        log_divide_2 = logexp.log(neg_data_mean / (1 - ratio))
        loss = (data_mean * log_divide_1 + neg_data_mean * log_divide_2).sum()
        data_diff = log_divide_1 - log_divide_2
        data_diff *= self._weight * scale / data.shape[0]
        diff += data_diff
        return loss * self._weight * scale

AutoencoderRegularizeLayer = make_regularize_layer(AutoencoderRegularizer)

        
