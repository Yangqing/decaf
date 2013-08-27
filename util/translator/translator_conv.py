from decaf.util.translator import conversions, registerer
from decaf.layers import core_layers
import numpy as np

def translator_conv(cuda_layer, output_shapes):
    if cuda_layer['groups'][0] != 1:
        raise ValueError('Group not supported yet.')
    num_kernels = cuda_layer['filters']
    ksize = cuda_layer['filterSize'][0]
    if not cuda_layer['sharedBiases']:
        raise ValueError('Unshared bias layers not supported yet.')
    stride = cuda_layer['stride'][0]
    pad = -cuda_layer['padding'][0]
    # figure out the output shape
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    padded_shape = (input_shape[0] + pad * 2,
                    input_shape[1] + pad * 2,
                    input_shape[2])
    output_shape = ((padded_shape[0] - ksize) / stride + 1,
                    (padded_shape[1] - ksize) / stride + 1,
                    num_kernels)
    output_shapes[cuda_layer['name']] = output_shape

    weight = cuda_layer['weights'][0]
    input_channels = cuda_layer['channels'][0]
    weight.resize((input_channels, ksize, ksize, num_kernels))
    converted_weight = np.empty((ksize, ksize, input_channels, num_kernels),
                                weight.dtype)
    for i in range(input_channels):
        converted_weight[:, :, i, :] = weight[i, :, :, :]
    converted_weight.resize(ksize * ksize * input_channels, num_kernels)
    #TODO: change the weights
    bias = cuda_layer['biases'].flatten().copy()
    decaf_layer = core_layers.ConvolutionLayer(
        name=cuda_layer['name'],
        num_kernels=num_kernels,
        ksize=ksize,
        stride=stride,
        pad=pad)
    param = decaf_layer.param()
    param[0].mirror(converted_weight)
    param[1].mirror(bias)
    return decaf_layer

registerer.register_translator('conv', translator_conv)
