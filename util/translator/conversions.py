"""Conversions converts data from the cuda convnet convention to the decaf
convention."""
import numpy as np



def imgs_cudaconv_to_decaf(imgs, size, channels, out=None):
    if out is None:
        out = np.empty((imgs.shape[0], size, size, channels), imgs.dtype)
    img_view = imgs.view()
    img_view.shape = (imgs.shape[0], channels, size, size)
    for i in range(channels):
        out[:, :, :, i] = img_view[:,i,:,:]
    return out

def img_cudaconv_to_decaf(img, size, channels, out=None):
    if out is None:
        out = np.empty((size, size, channels), img.dtype)
    return imgs_cudaconv_to_decaf(img[np.newaxis, :], size, channels,
                                  out=out[np.newaxis, :])[0]
