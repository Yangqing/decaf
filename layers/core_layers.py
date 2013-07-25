# pylint: disable=W0611
"""Imports commonly used layers."""

# Data Layers
from decaf.layers.ndarraydatalayer import NdarrayDataLayer

# Computation Layers
from decaf.layers.innerproduct import InnerProductLayer
from decaf.layers.loss import SquaredLossLayer, \
                              MultinomialLogisticLossLayer
