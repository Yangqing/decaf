"""Implements the LBFGS solver."""

from decaf import base
import logging
from scipy import optimize

_FMIN = optimize.fmin_l_bfgs_b

class LBFGSSolver(base.Solver):
    """The LBFGS solver."""
    
    def __init__(self, **kwargs):
        """The LBFGS solver. Necessary args is:
            lbfgs_args: a dictionary containg the parameters to be passed
                to lbfgs.
        """
        base.Solver.__init__(self, **kwargs)
        self._lbfgs_args = self.spec.get('lbfgs_args', {})
        self._param = None
        self._net = None

    def _collect_params(self, realloc=False):
        """Collect the network parameters into a long vector.
        """
        params_list = self._net.params()
        if self._param is None or realloc:
            total_size = sum(p.data.size for p in params_list)
            dtype = max(p.data.dtype for p in params_list)
            self._param = base.Blob(shape=total_size, dtype=dtype)
            self._param.init_diff()
        current = 0
        for param in params_list:
            size = param.data.size
            self._param.data[current:current+size] = param.data.flat
            self._param.diff[current:current+size] = param.diff.flat
            current += size

    def _distribute_params(self):
        """Distribute the parameter to the net.
        """
        params_list = self._net.params()
        current = 0
        for param in params_list:
            size = param.data.size
            param.data.flat = self._param.data[current:current+size]
            current += size

    def obj(self, variable):
        """The objective function that wraps around the net."""
        self._param.data[:] = variable
        self._distribute_params()
        loss = self._net.execute()
        self._collect_params()
        return loss, self._param.diff


    def solve(self, net):
        """Solves the net."""
        # first, run an execute pass to initialize all the parameters.
        self._net = net
        net.execute()
        self._collect_params(True)
        # now, run LBFGS
        # pylint: disable=W0108
        result = _FMIN(lambda x: self.obj(x), self._param.data, 
                       **self._lbfgs_args)
        # put the optimized result to the net.
        self._param.data[:] = result[0]
        self._distribute_params()
        logging.info("Final function value: %f.", result[1])

