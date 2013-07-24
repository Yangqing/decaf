from decaf import base
import logging
import numpy as np
from scipy import optimize

_FMIN = optimize.fmin_l_bfgs_b

class LBFGSSolver(base.Solver):
    def __init__(self, **kwargs):
        """The LBFGS solver. Necessary args is:
            lbfgs_args: a dictionary containg the parameters to be passed
                to lbfgs.
        """
        Solver.__init__(self, **kwargs)
        self._lbfgs_args = self.spec.get('lbfgs_args', {})
        self._param = None
        self._net = None

    def _collect_params(reload=False):
        """Collect the network parameters into a long vector.
        """
        params_list = self._net.params()
        if self._param is None or reload:
            total_size = sum(p.data.size for p in params_list)
            dtype = max(p.data.dtype for p in params_list)
            self._param = base.Blob(shape=total_size, dtype=dtype)
            self._param.init_diff()
        current = 0
        for param in param_list:
            size = param.data.size
            self._param.data[current:current+size] = param.data.flat
            self._param.diff[current:current+size] = param.diff.flat
            current += size

    def _distribute_params():
        """Distribute the parameter to the net.
        """
        param_list = self._net.params()
        current = 0
        for param in param_list:
            size = param.data.size
            param.data.flat = self._param.data[current:current+size]
            current += size

    @statimethod
    def obj(x, solver):
        solver._param.data[:] = x
        solver._distribute_params()
        loss = solver._net.execute()
        solver._collect_params()
        return loss, solver._param.diff


    def solve(net):
        """Solves the net."""
        # first, run an execute pass to initialize all the parameters.
        self._net = net
        net.execute()
        self._collect_params(True)
        # now, run LBFGS
        result = _FMIN(self.__class__.obj, self._param.data, 
                       args=[self], **self._lbfgs_args)
        # put the optimized result to the net.
        self._param.data[:] = result[0]
        self._distribute_params()
        logging.info("Final function value: %f.", result[1])

