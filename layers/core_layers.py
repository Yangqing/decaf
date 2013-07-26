# pylint: disable=W0611
"""Imports commonly used layers."""

# Data Layers
from decaf.layers.data.ndarraydata import NdarrayDataLayer
from decaf.layers.data.mnist import MNISTDataLayer

# Computation Layers
from decaf.layers.innerproduct import InnerProductLayer
from decaf.layers.loss import (SquaredLossLayer,
                              MultinomialLogisticLossLayer)
from decaf.layers.relu import ReLULayer

