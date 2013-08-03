# pylint: disable=W0611
"""Imports commonly used layers."""

# Data Layers
from decaf.layers.data.ndarraydata import NdarrayDataLayer
from decaf.layers.data.cifar import CifarDataLayer
from decaf.layers.data.mnist import MNISTDataLayer
from decaf.layers.sampler import (BasicMinibatchLayer,
                                  RandomPatchLayer)

# Computation Layers
from decaf.layers.innerproduct import InnerProductLayer
from decaf.layers.loss import (SquaredLossLayer,
                               MultinomialLogisticLossLayer)
from decaf.layers.relu import ReLULayer
from decaf.layers.split import SplitLayer
from decaf.layers.flatten import FlattenLayer
from decaf.layers.dropout import DropoutLayer
from decaf.layers.padding import PaddingLayer
from decaf.layers.sigmoid import SigmoidLayer
from decaf.layers.im2col import Im2colLayer
from decaf.layers.convolution import ConvolutionLayer
from decaf.layers.deconvolution import DeconvolutionLayer
from decaf.layers.pooling import PoolingLayer
