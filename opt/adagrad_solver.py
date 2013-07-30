"""Implements the LBFGS solver."""

from decaf import base
import logging
import numpy as np

class AdagradSolver(base.Solver):
    """The LBFGS solver."""
    
    def __init__(self, **kwargs):
        """The Adagrad solver. Necessary args are:
        
        base_lr: the base learning rate.
        base_accum: the base value to initialize the accumulated gradient
            diagonal. Default 1e-8.
        max_iter: the maximum number of iterations. Default 1000.
        """
        base.Solver.__init__(self, **kwargs)
        self._base_lr = self.spec['base_lr']
        self._base_accum = self.spec.get('base_accum', 1e-8)
        self._max_iter = self.spec.get('max_iter', 1000)
        self._accum = None

    def solve(self, decaf_net):
        """Solves the net."""
        # first, run a pass to initialize all the parameters.
        initial_loss = decaf_net.forward_backward()
        logging.info('Initial loss: %f.', initial_loss)
        # Initialize the accumulated matrix.
        self._accum = [np.empty(param.data().shape, param.data().dtype)
                       for param in decaf_net.params()]
        for accum in self._accum:
            accum[:] = self._base_accum
        # the main iteration
        logging.info('Adagrad started.')
        for iter_idx in range(self._max_iter):
            loss = decaf_net.forward_backward()
            gradient_scale = 0.
            # update gradient, and compute the diff
            for param, accum in zip(decaf_net.params(), self._accum):
                diff = param.diff()
                gradient_scale += np.dot(diff.flat, diff.flat)
                accum += diff * diff
                diff /= np.sqrt(accum)
                diff *= self._base_lr
            decaf_net.update()
            logging.info('Iter %d, f = %f, |g| = %f',
                         iter_idx, loss, np.sqrt(gradient_scale))
        logging.info('Adagrad finished.')
        logging.info('Final loss: %f.', loss)

