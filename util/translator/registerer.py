"""registerer is a simple module that allows one to register a custom
translator for a specific cuda layer.

== How to write a custom translator ==
Write your translate function in the format defined under translate_layer
below, and then register it with the type name of the corresponding cuda
convnet. Also, you need to import your module before the translation takes
place so your function actually gets registered.
"""

from decaf import base
from decaf.layers import core_layers
import logging

# OUTPUT_AFFIX is the affix we add to the layer name as the output blob name
# for the corresponding decaf layer.
OUTPUT_AFFIX = '_cudanet_out'
# _translators is a dictionary mapping layer names to functions that does the
# actual translations.
_translators = {}


def register_translator(name, translator):
    """Registers a translator."""
    _translators[name] = translator


def default_translator(cuda_layer, output_shapes):
    """A default translator if nothing fits: it will print a warning and then
    return a dummy base.Layer that does nothing.
    """
    logging.error('Default translator called.'
                  ' Will return a dummy layer for %s.', cuda_layer['name'])
    input_shape = output_shapes[cuda_layer['inputLayers'][0]['name']]
    output_shapes[cuda_layer['name']] = input_shape
    return core_layers.IdentityLayer(name=cuda_layer['name'])
    

def translate_layer(cuda_layer, output_shapes):
    """Translates a cuda layer to a decaf layer. The function will return
    False if the input layer is a data layer, in which no decaf layer needs to
    be inserted.

    Input:
        cuda_layer: a cuda layer as a dictionary, produced by the cuda convnet
            code.
        output_shapes: a dictionary keeping the output shapes of all the 
            layers.
    Output:
        decaf_layer: the corresponding decaf layer, or False if the input is a
            data layer.
    """
    if cuda_layer['type'] == 'data':
        # if the layer type is data, it is simply a data layer.
        return False
    layertype = cuda_layer['type']
    if layertype in _translators:
        return _translators[layertype](cuda_layer, output_shapes)
    else:
        return default_translator(cuda_layer, output_shapes)


def translate_cuda_network(cuda_layers, output_shapes):
    """Translates a list of cuda layers to a decaf net.

    Input:
        cuda_layers: a list of layers from the cuda convnet training.
        output_shapes: a dictionary that contains the specification on the
            input shapes. This dictionary will be modified in-place to add
            the output shapes for the intermediate layers, but you need to
            provide the shapes for all the data layers. For data that are
            going to be scalar, use -1. The shapes should be following the
            decaf convention, not the cuda convnet convention.
    """
    decaf_net = base.Net()
    for cuda_layer in cuda_layers:
        decaf_layer = translate_layer(cuda_layer, output_shapes)
        if not decaf_layer:
            # This is a data layer.
            logging.info('Considering %s as an input blob.', cuda_layer['name'])
            continue
        # Now, let's figure out the parent of the layer
        needs = []
        for idx in cuda_layer['inputs']:
            if cuda_layers[idx]['type'] == 'data':
                needs.append(cuda_layers[idx]['name'])
            else:
                needs.append(cuda_layers[idx]['name'] + OUTPUT_AFFIX)
        provide = cuda_layer['name'] + OUTPUT_AFFIX
        decaf_net.add_layers(decaf_layer, needs=needs, provides=provide)
    decaf_net.finish()
    return decaf_net

