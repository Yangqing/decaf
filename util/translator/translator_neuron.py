from decaf.util.translator import registerer
from decaf.layers import core_layers


def translator_neuron(cuda_layer, output_shapes):
    output_shapes[cuda_layer['name']] = output_shapes[cuda_layer['inputLayers'][0]['name']]
    neurontype = cuda_layer['neuron']['type']
    if neurontype == 'relu':
        return core_layers.ReLULayer(
            name=cuda_layer['name'])
    else:
        raise NotImplementedError('Neuron type %s not implemented yet.'
                                  % neurontype)

registerer.register_translator('neuron', translator_neuron)
