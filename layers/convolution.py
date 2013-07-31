"""Implements the convolution layer."""

from decaf import base
from decaf.layers import im2col, innerproduct, padding 

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
            memory: the approximate memory budget guideline (in bytes).
                This is used to determine how many intermediate storage we can
                keep. Default 1e7 (10 megabytes).
                CURRENTLY, THE MEMORY IS NOT USED AND THE CONVOLUTION ALWAYS
                RUNS PER-IMAGE.
        
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
        self._single_data = [base.Blob()]
        self._padded = [base.Blob()]
        self._col = [base.Blob()]
        self._conv_out = [base.Blob()]
        # set up the parameter - it's the same as the inner product param, but
        # we will have our own copy since the inner product param (especially
        # the diff) will be overwritten when we run on an per-image basis.
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
        self._ip_layer = innerproduct.InnerProductLayer(
            name=self.name + '_ip', num_output=self._num_kernels, bias=False)
    
    def forward(self, bottom, top):
        """Runs the forward pass."""
        # cache objects to avoid the [0] index.
        single_data = self._single_data[0]
        conv_out_blob = self._conv_out[0]
        bottom_data = bottom[0].data()
        if bottom_data.ndim == 3:
            # only one channel
            bottom_data.shape = bottom_data.shape + (1,)
        if not self._kernels.has_data():
            self._kernels.init_data((self._num_kernels, 
                                     self._ksize, 
                                     self._ksize, 
                                     bottom_data.shape[-1]),
                                    bottom_data.dtype)
        ip_kernel = self._ip_layer.param()[0].init_data(
            (self._ksize * self._ksize * bottom_data.shape[-1],
             self._num_kernels),
            self._kernels.data().dtype)
        ip_kernel.flat = self._kernels.data().flat
        # process data individually
        for i in range(bottom_data.shape[0]):
            single_data.mirror(bottom_data[i:i+1])
            self._pad_layer.forward(self._single_data, self._padded)
            self._im2col_layer.forward(self._padded, self._col)
            self._ip_layer.forward(self._col, self._conv_out)
            if i == 0:
                # initialize the top_data
                top_data = top[0].init_data((bottom_data.shape[0],) + \
                                            conv_out_blob.data().shape[1:],
                                            conv_out_blob.data().dtype)
            top_data[i] = conv_out_blob.data()
        return

    def backward(self, bottom, top, propagate_down):
        """Runs the backward pass."""
        single_data = self._single_data[0]
        conv_out_blob = self._conv_out[0]
        top_diff = top[0].diff()
        bottom_data = bottom[0].data()
        if bottom_data.ndim == 3:
            # only one channel
            bottom_data.shape = bottom_data.shape + (1,)
        kernel_diff = self._kernels.init_diff()
        if propagate_down:
            bottom_diff = bottom[0].init_diff()
        for i in range(bottom_data.shape[0]):
            # although it is a backward layer, we still need to compute
            # the intermediate results using forward calls.
            single_data.mirror(bottom_data[i:i+1])
            self._pad_layer.forward(self._single_data, self._padded)
            self._im2col_layer.forward(self._padded, self._col)
            conv_out_blob.mirror_diff(top_diff[i:i+1])
            self._ip_layer.backward(self._col, self._conv_out, propagate_down)
            kernel_diff.flat += self._ip_layer.param()[0].diff().flatten()
            if propagate_down:
                self._im2col_layer.backward(self._padded, self._col, True)
                self._pad_layer.backward(self._single_data, self._padded, True)
                bottom_diff[i].flat = single_data.diff().flat
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
        self._conv_out = [base.Blob()]
        return self.__dict__

    def update(self):
        """updates the parameters."""
        # Only the inner product layer needs to be updated.
        self._kernels.update()

