"""Implements basic fillers."""

from decaf import base
import numpy as np


# pylint: disable=R0903
class ConstantFiller(base.Filler):
    """Fills the values with a constant value.
    
    kwargs:
        value: the constant value to fill.
    """
    def fill(self, mat):
        """The fill function."""
        mat[:] = self.spec['value']


# pylint: disable=R0903
class RandFiller(base.Filler):
    """Fills the values with random numbers in [min, max).
    
    kwargs:
        min: the min value (default 0).
        max: the max value (default 1).
    """
    def fill(self, mat):
        """The fill function."""
        minval = self.spec.get('min', 0)
        maxval = self.spec.get('max', 1)
        mat[:] = np.random.random_sample(mat.shape)
        mat *= maxval - minval
        mat += minval

# pylint: disable=R0903
class RandIntFiller(base.Filler):
    """Fills the values with random numbers in [min, max).
    
    kwargs:
        low: the min value (default 0).
        high: the max value. Must be given.
    """
    def fill(self, mat):
        """The fill function."""
        lowval = self.spec.get('low', 0)
        highval = self.spec['high']
        mat[:] = np.random.randint(low=lowval, high=highval, size=mat.shape)

# pylint: disable=R0903
class GaussianRandFiller(base.Filler):
    """Fills the values with random gaussian.
    
    kwargs:
        mean: the mean value (default 0).
        std: the standard deviation (default 1).
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
    """Fills the values with boolean.

    kwargs:
        ratio: the ratio of 1 values when generating random binaries.
    """
    def fill(self, mat):
        """The fill function."""
        ratio = self.spec['ratio']
        mat[:] = np.random.random_sample(mat.shape) < ratio
