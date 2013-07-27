"""Implements basic fillers."""

from decaf import base
import numpy as np


# pylint: disable=R0903
class ConstantFiller(base.Filler):
    """Fills the values with a constant value.
    
    specs: value
    """
    def fill(self, mat):
        """The fill function."""
        mat[:] = self.spec['value']


# pylint: disable=R0903
class RandFiller(base.Filler):
    """Fill the values with random numbers in [min, max).
    
    Specs: min, max.
    """
    def fill(self, mat):
        """The fill function."""
        min = self.spec.get('min', 0)
        max = self.spec.get('max', 1)
        mat[:] = np.random.random_sample(mat.shape)
        mat *= max - min
        mat += min


# pylint: disable=R0903
class GaussianRandFiller(base.Filler):
    """Fill the values with random gaussian.
    
    Specs: mean, std.
    """
    def fill(self, mat):
        """The fill function."""
        mean = self.spec.get('mean', 0.)
        std = self.spec.get('std', 1.)
        mat[:] = np.random.standard_normal(mat.shape)
        mat *= std
        mat += mean

# pylint: disable=R0903
class DropoutFiller(base.Filler):
    """Fill the values with boolean.

    Specs: ratio: the ratio of 1 values.
    """
    def fill(self, mat):
        """The fill function."""
        ratio = self.spec['ratio']
        mat[:] = np.random.random_sample(mat.shape) < ratio
