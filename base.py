"""base.py implements the basic data types.
"""

from collections import defaultdict
import networkx as nx
import numpy as np


class DecafError(Exception):
    """NOOOOOOO! I need caffeine!
    
    Yes, this is the basic error type under decaf.
    """
    pass


class InvalidLayerError(DecafError):
    """The error when an invalid spec is passed to a layer."""
    pass


class InvalidNetError(DecafError):
    """The error raised when the network does not pass validation."""
    pass


class Filler(object):
    """This is the class that implements util functions to fill a blob.
    
    A filler implements the fill() function that takes a blob as the input,
    and fills the blob's data() field.
    """

    def __init__(self, **kwargs):
        """simply get the spec."""
        self.spec = kwargs

    def fill(self, mat):
        raise NotImplementedError


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
    def __init__(self, shape=None, dtype=np.float64, filler=None):
        self._data = None
        self._diff = None
        self._filler = filler
        if shape is not None and dtype is not None:
            self.init_data(shape, dtype)

   
    def mirror(self, input_array, shape=None):
        """Create the data as a view of the input array. This is useful to
        save space and avoid duplication for data layers.
        """
        self._data = input_array.view()
        if shape is not None:
            self._data.shape = shape
    
    def has_data(self):
        """Checks if the blob has data."""
        return self._data is not None
    
    def data(self):
        """Returns a view of the data."""
        return self._data.view()

    def has_diff(self):
        """Checks if the blob has diff."""
        return self._diff is not None

    def diff(self):
        """Returns the diff."""
        return self._diff.view()

    def update(self):
        """Update the data field by adding diff to it."""
        self._data += self._diff

    def init_data(self, shape, dtype=np.float64):
        """Initialize the data matrix if necessary."""
        if self.has_data() and self._data.shape == shape and \
           self._data.dtype == dtype:
            self._data[:] = 0
        else:
            self._data = np.zeros(shape, dtype)
        if self._filler is not None:
            self._filler.fill(self._data)
        return self.data()

    def init_diff(self):
        """Initialize the diff in the same format as data.
        
        Returns diff for easy access.
        """
        if not self.has_data():
            raise ValueError('The data should be initialized first!')
        if self.has_diff() and self._diff.shape == self._data.shape and \
           self._diff.dtype == self._data.dtype:
            self._diff[:] = 0
        else:
            self._diff = np.zeros(self._data.shape, self._data.dtype)
        return self.diff()


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
        self.freeze = self.spec.get('freeze', False)
        self._param = []

    def forward(self, bottom, top):
        """Computes the forward pass.
        
        Input:
            bottom: the data at the bottom.
            top: the top-layer output.
        """
        raise NotImplementedError

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass.
        Input:
            bottom: the data at the bottom.
            top: the data at the top.
            propagate_down: if set False, the gradient w.r.t. the bottom
                blobs does not need to be computed.
        Output:
            loss: the loss being generated in this layer. Note that if your
                layer does not generate any loss, you should still return 0.
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

    def backward(self, bottom, top, propagate_down):
        """No gradient needs to be computed for data.
        
        You should not override this function.
        """
        raise DecafError('You should not reach this.')

    def update(self):
        """The data layer has no parameter, and the update() function
        should not be called.
        """
        pass


# pylint: disable=R0921
class LossLayer(Layer):
    """A Layer that implements loss. Usually, the forward pass of the loss
    does the actual computation of both the loss and the gradients, and the
    backward pass will simply return the loss value. The loss layer should not
    accept any blobs on its top.
    """

    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self._loss = 0

    def forward(self, bottom, top):
        raise NotImplementedError

    def backward(self, bottom, top, propagate_down):
        return self._loss

    def update(self):
        pass


class Solver(object):
    """This is the very basic form of the solver."""
    def __init__(self, **kwargs):
        self.spec = kwargs

    def solve(self, net):
        """The solve function takes a net as an input, and optimizes its
        parameters.
        """
        raise NotImplementedError


class Regularizer(object):
    """This is the class that implements the regularization terms."""

    def __init__(self, **kwargs):
        """Initializes a regularizer. A regularizer needs a necessary keyword
        'weight'.
        """
        self.spec = kwargs
        self._weight = self.spec['weight']

    def reg(self, blob, num_data):
        """Compute the regularization term from the blob's data field, and
        add the regularization term to its diff directly.
        """
        raise NotImplementedError


class Net(object):
    """A Net is a directed graph with layer names and layer instances."""

    def __init__(self):
        self._graph = nx.DiGraph()
        self._blobs = defaultdict(Blob)
        self._layers = {}
        self._needs = {}
        self._provides = {}
        # The topological order to execute the layer.
        self._forward_order = None
        self._backward_order = None
        self._params = None
        self._finished = False

    def add_layer(self, layer, needs=None, provides=None):
        """Add a layer to the current network.

        Args:
            layer: a decaf.base.Layer instance.
            needs: a tuple of strings, indicating the blobs that the layer
                needs as its input.
            provides: similar to needs, but the layer's output instead.
        """
        # validate input
        if needs is None:
            needs = []
        if provides is None:
            provides = []
        if type(needs) is str:
            needs = [needs]
        if type(provides) is str:
            provides = [provides]
        self._finished = False
        # Add the layer
        if layer.name in self._layers or layer.name in self._blobs:
            raise InvalidNetError('A name already exists: %s' % layer.name)
        self._layers[layer.name] = layer
        # Add the blobs
        for blobname in needs + provides:
            if blobname in self._layers:
                raise InvalidNetError(
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
        #   propagate_down: whether the gradient w.r.t. to the bottom layer
        #       needs to be carried out.
        for name in topological_order:
            pred_need_backward = any(self._graph[p]['need_backward']
                                     for p in self._graph.predecessors(name))
            if name in self._layers:
                # see if a layer needs backward operation. A layer needs
                # backward operation if (1) it has parameters and isn't frozen
                # or (2) any of its predecessors needs backward operation.
                layer = self._layers[name]
                if (pred_need_backward or
                    (len(layer.param()) > 0 and not layer.freeze)):
                    self._graph[name]['need_backward'] = True
                else:
                    self._graph[name]['need_backward'] = False
                # see if a layer needs to compute its bottom diff. A layer
                # needs to compute its bottom diff if any of its predecessors
                # needs backward operation.
                if pred_need_backward:
                    self._graph[name]['propagate_down'] = True
                else:
                    self._graph[name]['propagate_down'] = False
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
                  self._graph[n]['propagate_down'])
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
            raise InvalidNetError('The network is not a DAG.')
        for blobname in self._blobs:
            # check if every blob has predecessors, and each predecessor is
            # a valid layer.
            predecessors = self._graph.predecessors(blobname)
            if len(predecessors) != 1:
                raise InvalidNetError(
                    'Blob %s has no source layer or multiple source layers.'
                    % blobname)
            if predecessors[0] not in self._layers:
                # TODO(Yangqing): Can this actually happen?
                raise InvalidNetError(
                    'Blob %s has a source that is not a layer.' % blobname)
            successors = self._graph.successors(blobname)
            if len(successors) > 1:
                # TODO(Yangqing): Maybe we would like to actually allow a blob
                # to have multiple successors?
                raise InvalidNetError(
                    'Blob %s has multiple successors.' % blobname)
        # TODO(Yangqing): maybe add validation for layers as well?
        # In the end, we will simply return True as a courtesy.
        return True
    
    def execute(self):
        """Execute one round of the networkx."""
        # the forward pass. We will also accumulate the loss function.
        if not self._finished:
            # Trying to modify an already finished network.
            raise DecafError('Call finish() before you use the network.')
        loss = 0.
        for _, layer, bottom, top in self._forward_order:
            layer.forward(bottom, top)
        # the backward pass
        for _, layer, bottom, top, propagate_down in self._backward_order:
            loss += layer.backward(bottom, top, propagate_down)
        return loss

    def update(self):
        """Update the parameters using the diff values provided in the
        parameters blob."""
        for _, layer, _, _ in self._forward_order:
            layer.update()
