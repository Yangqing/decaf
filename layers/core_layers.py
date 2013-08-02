# pylint: disable=W0611
"""Imports commonly used layers."""

# Data Layers
from decaf.layers.data.ndarraydata import NdarrayDataLayer
from decaf.layers.data.cifar import CifarDataLayer
from decaf.layers.data.mnist import MNISTDataLayer
from decaf.layers.sampler import BasicMinibatchLayer

# Computation Layers
from decaf.layers.innerproduct import InnerProductLayer
from decaf.layers.loss import (SquaredLossLayer,
                              MultinomialLogisticLossLayer)
from decaf.layers.relu import ReLULayer
from decaf.layers.split import SplitLayer
from decaf.layers.flatten import FlattenLayer
from decaf.layers.dropout import DropoutLayer
from decaf.layers.convolution import ConvolutionLayer
from decaf.layers.deconvolution import DeconvolutionLayer
