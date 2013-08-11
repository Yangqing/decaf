"""Utility to perform gradient check using scipy's check_grad method. 
"""

import numpy as np
from scipy import optimize


def _blobs_to_vec(blobs):
    """Collect the network parameters into a long vector.

    This method is not memory efficient - do NOT use in codes that require
    speed and memory.
    """
    if len(blobs) == 0:
        return np.array(())
    return np.hstack([blob.data().flatten() for blob in blobs])

def _blobs_diff_to_vec(blobs):
    """Similar to _blobs_to_vec, but copying diff."""
    if len(blobs) == 0:
        return np.array(())
    return np.hstack([blob.diff().flatten() for blob in blobs])

def _vec_to_blobs(vec, blobs):
    """Distribute the values in the vec to the blobs.
    """
    current = 0
    for blob in blobs:
        size = blob.data().size
        blob.data().flat = vec[current:current+size]
        current += size

def _vec_to_blobs_diff(vec, blobs):
    """Distribute the values in the vec to the blobs' diff part.
    """
    current = 0
    for blob in blobs:
        size = blob.diff().size
        blob.diff().flat = vec[current:current+size]
        current += size


class GradChecker(object):
    """A gradient checker that utilizes scipy.optimize.check_grad to perform
    the gradient check.

    The gradient checker checks the gradient with respect to both the params
    and the bottom blobs if they exist. It checks 2 types of object functions:
        (1) the squared sum of all the outputs.
        (2) each of the output value.
    The total number of functions to be tested is (num of outputs + 1).

    The check is carried out by the check() function, which checks all the
    cases above. If any error exceeds the threshold, the check function will
    return a tuple: (False, index, err), where index is the index where error
    exceeds threshold, and err is the error value. index=-1 means the squared
    sum case. If all errors are under threshold, the check function will
    return (True, max_err) where max_err is the maximum error encountered.
    """

    def __init__(self, threshold):
        """Initializes the checker.

        Input:
            threshold: the threshold to reject the gradient value.
        """
        self._threshold = threshold
    
    @staticmethod
    def _func_net(x_init, decaf_net):
        """function wrapper for a net."""
        _vec_to_blobs(x_init, decaf_net.params())
        return decaf_net.forward_backward()

    @staticmethod
    def _grad_net(x_init, decaf_net):
        """gradient wrapper for a net."""
        _vec_to_blobs(x_init, decaf_net.params())
        decaf_net.forward_backward()
        return _blobs_diff_to_vec(decaf_net.params())

    @staticmethod
    def _func(x_init, layer, input_blobs, output_blobs, check_data, idx,
             checked_blobs):
        """The function. It returns the output at index idx, or if idx is
        negative, computes an overall loss by taking the squared sum of all
        output values.
        """
        if check_data:
            _vec_to_blobs(x_init, checked_blobs)
        else:
            _vec_to_blobs(x_init, layer.param())
        layer.forward(input_blobs, output_blobs)
        if len(output_blobs) > 0:
            output = _blobs_to_vec(output_blobs)
        else:
            # a dummy output
            output = np.array([0])
        for blob in output_blobs:
            blob.init_diff()
        # The layer may have reg terms, so we run a dummy backward
        additional_loss = layer.backward(input_blobs, output_blobs, True)
        if idx < 0:
            return np.dot(output, output) + additional_loss
        else:
            return output[idx] + additional_loss

    @staticmethod
    def _grad(x_init, layer, input_blobs, output_blobs, check_data, idx,
              checked_blobs):
        """The coarse gradient."""
        if check_data:
            _vec_to_blobs(x_init, checked_blobs)
        else:
            _vec_to_blobs(x_init, layer.param())
        layer.forward(input_blobs, output_blobs)
        # initialize the diff
        for blob in output_blobs:
            blob.init_diff()
        if len(output_blobs) > 0:
            output = _blobs_to_vec(output_blobs)
            if idx < 0:
                output *= 2.
            else:
                output[:] = 0
                output[idx] = 1.
            _vec_to_blobs_diff(output, output_blobs)
        # Now, get the diff
        if check_data:
            layer.backward(input_blobs, output_blobs, True)
            return _blobs_diff_to_vec(checked_blobs)
        else:
            layer.backward(input_blobs, output_blobs, False)
            return _blobs_diff_to_vec(layer.param())

    def check_network(self, decaf_net):
        """Checks a whole decaf network. Your network should not contain any
        stochastic components: multiple forward backward passes should produce
        the same value for the same parameters.
        """
        # Run a round to initialize the params.
        decaf_net.forward_backward()
        param_backup = _blobs_to_vec(decaf_net.params())
        x_init = param_backup.copy()
        # pylint: disable=E1101
        err = optimize.check_grad(GradChecker._func_net, GradChecker._grad_net,
                                  x_init, decaf_net)
        if err > self._threshold:
            return (False, err)
        else:
            return (True, err)

    def check(self, layer, input_blobs, output_blobs, check_indices = None):
        """Checks a layer with given input blobs and output blobs.
        """
        # pre-run to get the input and output shapes.
        if check_indices is None:
            checked_blobs = input_blobs
        else:
            checked_blobs = [input_blobs[i] for i in check_indices]
        layer.forward(input_blobs, output_blobs)
        input_backup = _blobs_to_vec(checked_blobs)
        param_backup = _blobs_to_vec(layer.param())
        num_output = _blobs_to_vec(output_blobs).size
        max_err = 0
        # first, check grad w.r.t. param
        x_init = _blobs_to_vec(layer.param())
        if len(x_init) > 0:
            for i in range(-1, num_output):
                # pylint: disable=E1101
                err = optimize.check_grad(
                    GradChecker._func, GradChecker._grad, x_init,
                    layer, input_blobs, output_blobs, False, i, checked_blobs)
                max_err = max(err, max_err)
                if err > self._threshold:
                    return (False, i, err, 'param')
            # restore param
            _vec_to_blobs(param_backup, layer.param())
        # second, check grad w.r.t. input
        x_init = _blobs_to_vec(checked_blobs)
        if len(x_init) > 0:
            for i in range(-1, num_output):
                # pylint: disable=E1101
                err = optimize.check_grad(
                    GradChecker._func, GradChecker._grad, x_init,
                    layer, input_blobs, output_blobs, True, i, checked_blobs)
                max_err = max(err, max_err)
                if err > self._threshold:
                    return (False, i, err, 'input')
            # restore input
            _vec_to_blobs(input_backup, checked_blobs)
        return (True, max_err)
