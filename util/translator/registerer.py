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


def default_translator(cuda_layer):
    """A default translator if nothing fits: it will print a warning and then
    return a dummy base.Layer that does nothing.
    """
    logging.error('Default translator called.'
                  ' Will return a dummy layer for %s.', cuda_layer['name'])
    return core_layers.IdentityLayer(name=cuda_layer['name'])
    

def translate_layer(cuda_layer):
    """Translates a cuda layer to a decaf layer. The function will return
    False if the input layer is a data layer, in which no decaf layer needs to
    be inserted.

    Input:
        cuda_layer: a cuda layer as a dictionary, produced by the cuda convnet
            code.
    Output:
        decaf_layer: the corresponding decaf layer, or False if the input is a
            data layer.
    """
    if cuda_layer['type'] == 'data':
        # if the layer type is data, it is simply a data layer.
        return False
    layertype = cuda_layer['type']
    if layertype in _translators:
        return _translators[layertype](cuda_layer)
    else:
        return default_translator(cuda_layer)


def translate_cuda_network(cuda_layers):
    """Translates a list of cuda layers to a decaf net.
    """
    decaf_net = base.Net()
    for cuda_layer in cuda_layers:
        decaf_layer = translate_layer(cuda_layer)
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

