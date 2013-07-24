"""Safe log and exp computation."""

import numpy as np


def exp(mat, out=None):
    """ A (hacky) safe exp that avoids overflowing
    Input:
        mat: the input ndarray
        out: (optional) the output ndarray. Could be in-place.
    Output:
        out: the output ndarray
    """
    if out is None:
        out = np.empty_like(mat)
    np.clip(mat, -np.inf, 100, out=out)
    np.exp(out, out=out)
    return out


def log(mat, out=None):
    """ A (hacky) safe log that avoids nans
    
    Note that if there are negative values in the input, this function does not
    throw an error. Handle these cases with care.
    """
    if out is None:
        out = np.empty_like(mat)
    # pylint: disable=E1101
    np.clip(mat, np.finfo(mat.dtype).tiny, np.inf, out=out)
    np.log(out, out=out)
    return out

