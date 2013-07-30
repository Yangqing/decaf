"""Implements the inner product layer."""

from decaf import base

class BasicMinibatchLayer(base.DataLayer):
    """A layer that extracts minibatches from bottom blobs.
    
    We will not randomly generate minibatches, but will instead produce them
    sequentially. Every forward() call will change the minibatch, so if you
    want a fixed minibatch, do NOT run forward multiple times.
    """

    def __init__(self, **kwargs):
        """Initializes the layer.
        """
        base.DataLayer.__init__(self, **kwargs)
        self._minibatch = self.spec['minibatch']
        self._index = 0

    def forward(self, bottom, top):
        """Computes the forward pass."""
        size = bottom[0].data().shape[0]
        end_id = self._index + self._minibatch
        for bottom_blob, top_blob in zip(bottom, top):
            bottom_data = bottom_blob.data()
            if bottom_data.shape[0] != size:
                raise RuntimeError(
                    'Inputs do not have identical number of data points!')
            top_data = top_blob.init_data(
                (self._minibatch,) + bottom_data.shape[1:], bottom_data.dtype)
            # copy data
            if end_id <= size:
                top_data[:] = bottom_data[self._index:end_id]
            else:
                top_data[:(size - self._index)] = bottom_data[self._index:]
                top_data[-(end_id - size):] = bottom_data[:(end_id - size)]
        # finally, compute the new index.
        self._index = end_id % size
        
    def update(self):
        """ReLU has nothing to update."""
        pass
