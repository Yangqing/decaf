"""Implements the convolution layer."""

from decaf import base
from decaf.layers import im2col, innerproduct, padding 
from decaf.util import blasdot
import numpy as np

class ConvolutionLayer(base.Layer):
    """A layer that implements the convolution function."""

    def __init__(self, **kwargs):
        """Initializes the convolution layer. Strictly, this is a correlation
        layer since the kernels are not reversed spatially as in a classical
        convolution operation.

        kwargs:
            name: the name of the layer.
            num_kernels: the number of kernels.
            ksize: the kernel size. Kernels will be square shaped and have the
                same number of channels as the data.
            stride: the kernel stride.
            mode: 'valid', 'same', or 'full'.
            reg: the regularizer to be used to add regularization terms.
                should be a decaf.base.Regularizer instance. Default None. 
            filler: a filler to initialize the weights. Should be a
                decaf.base.Filler instance. Default None.
        
        When computing convolutions, we will always start from the top left
        corner, and any rows/columns on the right and bottom sides that do not
        fit the stride will be discarded. To enforce the 'same' mode to return
        results of the same size as the data, we require the 'same' mode to be
        paired with an odd number as the kernel size.
        """
        base.Layer.__init__(self, **kwargs)
        self._num_kernels = self.spec['num_kernels']
        self._ksize = self.spec['ksize']
        self._stride = self.spec['stride']
        self._mode = self.spec['mode']
        self._reg = self.spec.get('reg', None)
        self._filler = self.spec.get('filler', None)
        self._memory = self.spec.get('memory', 1e7)
        if self._ksize <= 1:
            raise ValueError('Invalid kernel size. Kernel size should > 1.')
        if self._mode == 'same' and self._ksize % 2 == 0:
            raise ValueError('The "same" mode should have an odd kernel size.')
        # since the im2col operation often creates large intermediate matrices,
        # we will have intermediate blobs to store them.
        self._single_data = base.Blob()
        self._padded = base.Blob()
        self._col = base.Blob()
        # set up the parameter
        self._kernels = base.Blob(filler=self._filler)
        self._param = [self._kernels]
        # Constructs the sub layers that actually carry out the convolution.
        if self._mode == 'valid':
            pad = 0
        elif self._mode == 'full':
            pad = self._ksize - 1
        elif self._mode == 'same':
            pad = int(self._ksize / 2)
        else:
            raise ValueError('Unknown mode: %s' % self._mode)
        # construct the layers
        self._pad_layer = padding.PaddingLayer(
            name=self.name + '_pad', pad = pad)
        self._im2col_layer = im2col.Im2colLayer(
            name=self.name + '_im2col',
            psize=self._ksize,
            stride=self._stride)
    
    def forward(self, bottom, top):
        """Runs the forward pass."""
        single_data = self._single_data
        bottom_data = bottom[0].data()
        if bottom_data.ndim == 3:
            # only one channel
            bottom_data.shape = bottom_data.shape + (1,)
        if not self._kernels.has_data():
            # initialize the kernels
            self._kernels.init_data(
                (self._ksize * self._ksize * bottom_data.shape[-1],
                 self._num_kernels),
                bottom_data.dtype)
        # process data individually
        for i in range(bottom_data.shape[0]):
            # mirror input
            single_data.mirror(bottom_data[i:i+1])
            # pad
            self._pad_layer.forward([self._single_data], [self._padded])
            # call im2col
            self._im2col_layer.forward([self._padded], [self._col])
            if i == 0:
                # initialize the top_data
                top_data = top[0].init_data(
                    (bottom_data.shape[0],
                     self._col.data().shape[1],
                     self._col.data().shape[2],
                     self._num_kernels),
                    bottom_data.dtype)
            # inner product
            blasdot.dot_lastdim(self._col.data()[0], self._kernels.data(),
                                out=top_data[i])
        return

    def backward(self, bottom, top, propagate_down):
        """Runs the backward pass."""
        single_data = self._single_data
        top_diff = top[0].diff()
        bottom_data = bottom[0].data()
        if bottom_data.ndim == 3:
            # only one channel
            bottom_data.shape = bottom_data.shape + (1,)
        kernel_diff = self._kernels.init_diff()
        kernel_diff_buffer = np.zeros_like(kernel_diff)
        if propagate_down:
            bottom_diff = bottom[0].init_diff()
        for i in range(bottom_data.shape[0]):
            # although it is a backward layer, we still need to compute
            # the intermediate results using forward calls.
            single_data.mirror(bottom_data[i:i+1])
            self._pad_layer.forward([self._single_data], [self._padded])
            self._im2col_layer.forward([self._padded], [self._col])
            col_data = self._col.data()[0]
            blasdot.dot_firstdims(col_data, top_diff[i],
                                 out=kernel_diff_buffer)
            kernel_diff += kernel_diff_buffer
            if propagate_down:
                single_data.mirror_diff(bottom_diff[i:i+1])
                col_diff = self._col.init_diff()[0]
                blasdot.dot_lastdim(top_diff[i], self._kernels.data().T,
                                    out=col_diff)
                # im2col backward
                self._im2col_layer.backward([self._padded], [self._col], True)
                # pad backward
                self._pad_layer.backward([self._single_data], [self._padded], True)
        # finally, add the regularization term
        if self._reg is not None:
            return self._reg.reg(self._kernels, bottom_data.shape[0])
        else:
            return 0.

    def __getstate__(self):
        """When pickling, we will remove the intermediate data."""
        self._single_data = [base.Blob()]
        self._padded = [base.Blob()]
        self._col = [base.Blob()]
        return self.__dict__

    def update(self):
        """updates the parameters."""
        # Only the inner product layer needs to be updated.
        self._kernels.update()

