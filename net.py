"""Defines the basic structure of a net."""

from collections import defaultdict
from decaf import base
import networkx as nx


class InvalidNetworkError(base.DecafError):
    """The error raised when the network does not pass validation."""
    pass


class Net(object):
    """A Net is a directed graph with layer names and layer instances."""

    def __init__(self):
        self._graph = nx.DiGraph()
        self._blobs = defaultdict(base.Blob)
        self._layers = {}
        self._needs = {}
        self._provides = {}
        # The topological order to execute the layer.
        self._forward_order = None
        self._backward_order = None
        self._params = None
        self._finished = False


    def add_layer(self, layer, needs=(), provides=()):
        """Add a layer to the current network.

        Args:
            layer: a decaf.base.Layer instance.
            needs: a tuple of strings, indicating the blobs that the layer
                needs as its input.
            provides: similar to needs, but the layer's output instead.
        """
        if self._finished:
            # Trying to modify an already finished network.
            raise base.DecafError('Modifying an already finished net.')
        # Add the layer
        if layer.name in self._layers:
            raise InvalidNetworkError('Duplicated layer found: %s' % layer.name)
        if layer.name in self._blobs:
            raise InvalidNetworkError(
                'Layer name found as a blob: %s' % layer.name)
        self._layers[layer.name] = layer
        # Add the blobs
        for blobname in needs:
            if blobname in self._layers:
                raise InvalidNetworkError(
                    'Blob name found as a layer: %s' % blobname)
        for blobname in provides:
            if blobname in self._layers:
                raise InvalidNetworkError(
                    'Blob name found as a layer: %s' % blobname)
        self._needs[layer.name] = [self._blobs[blobname] for blobname in needs]
        self._provides[layer.name] = [self._blobs[blobname]
                                      for blobname in provides]
        # create the graph structure
        for blobname in needs:
            self._graph.add_edge(blobname, layer.name)
        for blobname in provides:
            self._graph.add_edge(layer.name, blobname)
   
    def finish(self):
        """Call this function when you finish the network construction."""
        # validate.
        self._validate()
        topological_order = nx.topological_sort(self._graph)
        # For efficiency reasons, we will see for each layer, whether the
        # backward operation needs to be carried out.
        # This is stored in two parameters:
        #   need_backward: whether the backward pass needs to be carried out.
        #   need_bottom_diff: whether the gradient w.r.t. to the bottom layer
        #       needs to be carried out.
        for name in topological_order:
            pred_need_backward = any(self._graph[p]['need_backward']
                                     for p in self._graph.predecessors(name))
            if name in self._layers:
                # see if a layer needs backward operation. A layer needs
                # backward operation if (1) it has parameters, or (2) any of
                # its predecessors needs backward operation.
                if self._layers[name].param() or pred_need_backward:
                    self._graph[name]['need_backward'] = True
                else:
                    self._graph[name]['need_backward'] = False
                # see if a layer needs to compute its bottom diff. A layer
                # needs to compute its bottom diff if any of its predecessors
                # needs backward operation.
                if pred_need_backward:
                    self._graph[name]['need_bottom_diff'] = True
                else:
                    self._graph[name]['need_bottom_diff'] = False
            else:
                # see if a blob needs backward operation.
                # This is only used so we can verify further layers.
                self._graph[name]['need_backward'] = pred_need_backward
        # create the order to run forward and backward passes
        layerorder = [layername for layername in topological_order
                      if layername in self._layers]
        self._forward_order = \
                [(n, self._layers[n], self._needs[n], self._provides[n])
                 for n in layerorder]
        self._backward_order = \
                [(n, self._layers[n], self._needs[n], self._provides[n],
                  self._graph[n]['need_bottom_diff'])
                 for n in layerorder[::-1]
                 if self._graph[n]['need_backward']]
        # store all the parameters
        self._params = []
        for name in layerorder:
            self._params.extend(self._layers[name].param())
        # Note: Any further finishing code should be inserted here.
        self._finished = True
    
    def params(self):
        """Return a list of parameters used in the network."""
        return self._params

    def _validate(self):
        """Validates if a network is executable. A network being executable
        means that every blob node has a layer as its predecessor, and no loop
        exists in the network.
        """
        if not nx.is_directed_acyclic_graph(self._graph):
            raise InvalidNetworkError('The network is not a DAG.')
        for blobname in self._blobs:
            # check if every blob has predecessors, and each predecessor is
            # a valid layer.
            predecessors = self._graph.predecessors(blobname)
            if len(predecessors) != 1:
                raise InvalidNetworkError(
                    'Blob %s has no source layer or multiple source layers.'
                    % blobname)
            if predecessors[0] not in self._layers:
                # TODO(Yangqing): Can this actually happen?
                raise InvalidNetworkError(
                    'Blob %s has a source that is not a layer.' % blobname)
            successors = self._graph.successors(blobname)
            if len(successors) > 1:
                # TODO(Yangqing): Maybe we would like to actually allow a blob
                # to have multiple successors?
                raise InvalidNetworkError(
                    'Blob %s has multiple successors.' % blobname)
        # TODO(Yangqing): maybe add validation for layers as well?
        # In the end, we will simply return True as a courtesy.
        return True
    
    def execute(self):
        """Execute one round of the networkx."""
        # the forward pass. We will also accumulate the loss function.
        loss = 0.
        for _, layer, bottom, top in self._forward_order:
            loss += layer.forward(bottom, top)
        # the backward pass
        for _, layer, bottom, top, need_bottom_diff in self._backward_order:
            layer.backward(bottom, top, need_bottom_diff)
        return loss

    def update(self):
        """Update the parameters using the diff values provided in the
        parameters blob."""
        for _, layer, _, _ in self._forward_order:
            layer.update()
