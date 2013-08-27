from decaf.util.translator import registerer
from decaf.layers import core_layers


def translator_cmrnorm(cuda_layer, output_shapes):
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    output_shapes[cuda_layer['name']] = input_shape
    return core_layers.LocalResponseNormalizeLayer(
        name=cuda_layer['name'],
        size=cuda_layer['size'],
        k=1, # TODO: check if this is the actual hard-coded value for k
        alpha = cuda_layer['scale'],
        beta = cuda_layer['pow'])

registerer.register_translator('cmrnorm', translator_cmrnorm)
