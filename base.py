"""base.py implements the basic data types.
"""

import logging
import numpy as np


class DecafError(Exception):
    """NOOOOOOO! I need caffeine!
    
    Yes, this is the basic error type under decaf.
    """
    pass

class InvalidSpecError(DecafError):
    """The error when an invalid spec is passed to a layer."""
    pass


# pylint: disable=R0903
class Blob(object):
    """Blob is the data structure that holds a piece of numpy array as well as
    its gradient so that we can accumulate and pass around data more easily.

    We define two numpy matrices: one is data, which stores the data in the
    current blob; the other is diff (short for difference): when a network
    runs its forward and backward pass, diff will store the gradient value;
    when a solver goes through the blobs, diff will then be replaced with the
    value to update.

    The diff matrix will not be created unless you explicitly run init_diff,
    as many Blobs do not need the gradients to be computed.
    """
    def __init__(self, shape=None, dtype=None):
        if shape is None and dtype is None:
            self.data = None
        else:
            self.data = np.zeros(shape, dtype=dtype)
        self.diff = None
   
    def mirror(input_array):
        """Create the data as a view of the input array. This is useful to
        save space and avoid duplication for data layers.
        """
        self.data = input_array.view()

    def resize(shape, dtype):
        """Ensures that the blob has the right data. If not, simply change
        the size of the data matrix.."""
        if self.data.shape == shape and self.data.dtype == dtype:
            pass
        else:
            # Blob resize should not happen often. If it happens, we log it
            # so the user can know if multiple resizes take place.
            logging.info('Blob resized to %s, dtype %s',
                         str(shape), str(dtype))
            self.data = np.zeros(shape, dtype=dtype)

    def init_diff(self):
        """Initialize the diff in the same format as data.
        
        Returns diff for easy access.
        """
        if self.diff and self.diff.shape == self.data.shape and \
           self.diff.dtype == self.data.dtype:
            self.diff[:] = 0
        else:
            self.diff = np.zeros_like(self.data)
        return diff


class Layer(object):
    """A Layer is the most basic component in decal. It takes multiple blobs
    as its input, and produces its outputs as multiple blobs. The parameter
    to be learned in the layers are 
    
    When designing layers, always make sure that your code deals with minibatches.
    """

    def __init__(self, **kwargs):
        """Creates a Layer.

        Necessary argument:
            name: the name of the layer.
        """
        self.spec = kwargs
        self.name = self.spec['name']
        self._param = []

    def forward(self, bottom, top):
        """Computes the forward pass.
        
        Input:
            bottom: the data at the bottom.
            top: the top-layer output.
        Output:
            loss: the loss being generated in this layer. Note that if your
                layer does not generate any loss, you should still return 0.
        """
        raise NotImplementedError

    def backward(self, bottom, top, need_bottom_diff):
        """Computes the backward pass.
        Input:
            bottom: the data at the bottom.
            top: the data at the top.
            need_bottom_diff: if set False, the gradient w.r.t. the bottom
                blobs does not need to be computed.
        """
        raise NotImplementedError
    
    def update(self):
        """Updates my parameters, based on the diff value given in the param
        blob.
        """
        raise NotImplementedError

    def param(self):
        """Returns the parameters in this layer. It should be a list of
        Blob objects.
        
        In our layer, either collect all your parameters into the self._param
        list, or implement your own param() function.
        """
        return self._param


# pylint: disable=R0921
class DataLayer(Layer):
    """A Layer that generates data.
    """
    
    def forward(self, bottom, top):
        """Generates the data.
        
        Your data layer should override this function.
        """
        raise NotImplementedError

    def backward(self, bottom, top, need_bottom_diff):
        """No gradient needs to be computed for data.
        
        You should not override this function.
        """
        raise base.DecafError('You should not reach this.')
        return None

    def update(self):
        """The data layer has no parameter, and the update() function
        should not be called.
        """
        raise base.DecafError('You should not reach this.')


# pylint: disable=R0921
class LossLayer(Layer):
    """A Layer that implements loss. The forward pass of the loss layer will
    produce the loss, and the backward pass will compute the gradient based on
    the network output and the training data.
    """
    def forward(self, bottom, top):
        raise NotImplementedError

    def backward(self, bottom, top, need_bottom_diff):
        raise NotImplementedError

    def update(self):
        pass

    def param(self):
        return []


class Solver(object):
    """This is the very basic form of the solver."""
    def __init__(self, **kwargs):
        self.spec = kwargs

    def solve(net):
        """The solve function takes a net as an input, and optimizes its
        parameters.
        """
        raise NotImplementedError

