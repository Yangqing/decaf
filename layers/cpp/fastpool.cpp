#include <cstring>
#include <cmath>
#include <algorithm>

#include "fastpool.h"

using std::max;
using std::min;

template <typename Dtype>
inline void maxpooling_forward(
        const Dtype* image, Dtype* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = (height - psize) / stride + 1;
    int pooled_width = (width - psize) / stride + 1;
    int last_height = stride * (pooled_height - 1) + psize;
    int last_width = stride * (pooled_width - 1) + psize;
    memset(pooled, 0, sizeof(Dtype) * pooled_height * pooled_width * nchannels);
    // This code is written in a forward mode: we go through the pixels once,
    // and write to all the pooled regions that it maps to.
    for (int i = 0; i < last_height; ++i) {
        for (int j = 0; j < last_width; ++j) {
            // Processing pixel at [i,j].
            // First, compute the pooling region
            int h_start = (i < psize) ? 0 : (i - psize) / stride + 1;
            int h_end = min(i / stride + 1, pooled_height);
            int w_start = (j < psize) ? 0 : (j - psize) / stride + 1;
            int w_end = min(j / stride + 1, pooled_width);
            const Dtype* p_image = image + (i * width + j) * nchannels;
            for (int ph = h_start; ph < h_end; ++ph) {
                for (int pw = w_start; pw < w_end; ++pw) {
                    Dtype* p_pooled = pooled + (ph * pooled_width + pw) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        p_pooled[c] = max(p_pooled[c], p_image[c]);
                    }
                }
            }
        } // loop over width
    } // loop over height
}


template <typename Dtype>
inline void maxpooling_backward(
        const Dtype* image, const Dtype* pooled, Dtype* image_grad,
        const Dtype* pooled_grad, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = (height - psize) / stride + 1;
    int pooled_width = (width - psize) / stride + 1;
    int last_height = stride * (pooled_height - 1) + psize;
    int last_width = stride * (pooled_width - 1) + psize;
    memset(image_grad, 0, sizeof(Dtype) * height * width * nchannels);
    // This code is written in a forward mode: we go through the pixels once,
    // and write to all the pooled regions that it maps to.
    for (int i = 0; i < last_height; ++i) {
        for (int j = 0; j < last_width; ++j) {
            // Processing pixel at [i,j].
            // First, compute the pooling region
            int h_start = (i < psize) ? 0 : (i - psize) / stride + 1;
            int h_end = min(i / stride + 1, pooled_height);
            int w_start = (j < psize) ? 0 : (j - psize) / stride + 1;
            int w_end = min(j / stride + 1, pooled_width);
            const Dtype* p_image = image + (i * width + j) * nchannels;
            Dtype* p_image_grad = image_grad + (i * width + j) * nchannels;
            for (int ph = h_start; ph < h_end; ++ph) {
                for (int pw = w_start; pw < w_end; ++pw) {
                    const Dtype* p_pooled = pooled + (ph * pooled_width + pw) * nchannels;
                    const Dtype* p_pooled_grad = pooled_grad + 
                        (ph * pooled_width + pw) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        if (p_image[c] + EQUAL_THRESHOLD > p_pooled[c]) {
                            p_image_grad[c] += p_pooled_grad[c];
                        }
                    }
                }
            }
        } // loop over width
    } // loop over height
}


template <typename Dtype>
inline void avepooling_forward(
        const Dtype* image, Dtype* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = (height - psize) / stride + 1;
    int pooled_width = (width - psize) / stride + 1;
    int last_height = stride * (pooled_height - 1) + psize;
    int last_width = stride * (pooled_width - 1) + psize;
    memset(pooled, 0, sizeof(Dtype) * pooled_height * pooled_width * nchannels);
    for (int i = 0; i < last_height; ++i) {
        for (int j = 0; j < last_width; ++j) {
            // Processing pixel at [i,j].
            // First, compute the pooling region
            int h_start = (i < psize) ? 0 : (i - psize) / stride + 1;
            int h_end = min(i / stride + 1, pooled_height);
            int w_start = (j < psize) ? 0 : (j - psize) / stride + 1;
            int w_end = min(j / stride + 1, pooled_width);
            const Dtype* p_image = image + (i * width + j) * nchannels;
            for (int ph = h_start; ph < h_end; ++ph) {
                for (int pw = w_start; pw < w_end; ++pw) {
                    Dtype* p_pooled = pooled + (ph * pooled_width + pw) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        p_pooled[c] += p_image[c];
                    }
                }
            }
        } // loop over width
    } // loop over height
    // do average
    int total = pooled_height * pooled_width * nchannels;
    Dtype scale = 1. / psize / psize;
    for (int i = 0; i < total; ++i) {
        pooled[i] *= scale;
    }
}

template <typename Dtype>
inline void avepooling_backward(
        Dtype* image_grad, const Dtype* pooled_grad, const int height, 
        const int width, const int nchannels, const int psize,
        const int stride) {
    int pooled_height = (height - psize) / stride + 1;
    int pooled_width = (width - psize) / stride + 1;
    int last_height = stride * (pooled_height - 1) + psize;
    int last_width = stride * (pooled_width - 1) + psize;
    memset(image_grad, 0, sizeof(Dtype) * height * width * nchannels);
    // This code is written in a forward mode: we go through the pixels once,
    // and write to all the pooled regions that it maps to.
    for (int i = 0; i < last_height; ++i) {
        for (int j = 0; j < last_width; ++j) {
            // Processing pixel at [i,j].
            // First, compute the pooling region
            int h_start = (i < psize) ? 0 : (i - psize) / stride + 1;
            int h_end = min(i / stride + 1, pooled_height);
            int w_start = (j < psize) ? 0 : (j - psize) / stride + 1;
            int w_end = min(j / stride + 1, pooled_width);
            Dtype* p_image_grad = image_grad + (i * width + j) * nchannels;
            for (int ph = h_start; ph < h_end; ++ph) {
                for (int pw = w_start; pw < w_end; ++pw) {
                    const Dtype* p_pooled_grad = pooled_grad + 
                        (ph * pooled_width + pw) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        p_image_grad[c] += p_pooled_grad[c];
                    }
                }
            }
        } // loop over width
    } // loop over height
    // do average
    int total = height * width * nchannels;
    Dtype scale = 1. / psize / psize;
    for (int i = 0; i < total; ++i) {
        image_grad[i] *= scale;
    }
}

extern "C" {

void maxpooling_forward_float(
        const float* image, float* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    maxpooling_forward<float>(image, pooled, height, width, nchannels, psize,
            stride);
}

void maxpooling_backward_float(
        const float* image, const float* pooled, float* image_grad,
        const float* pooled_grad, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    maxpooling_backward<float>(image, pooled, image_grad, pooled_grad,
            height, width, nchannels, psize, stride);
}

void avepooling_forward_float(
        const float* image, float* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    avepooling_forward<float>(image, pooled, height, width, nchannels, psize,
            stride);
}

void avepooling_backward_float(
        float* image_grad, const float* pooled_grad, const int height, 
        const int width, const int nchannels, const int psize,
        const int stride) {
    avepooling_backward<float>(image_grad, pooled_grad, height,width,
            nchannels, psize, stride);
}

void maxpooling_forward_double(
        const double* image, double* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    maxpooling_forward<double>(image, pooled, height, width, nchannels, psize,
            stride);
}

void maxpooling_backward_double(
        const double* image, const double* pooled, double* image_grad,
        const double* pooled_grad, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    maxpooling_backward<double>(image, pooled, image_grad, pooled_grad,
            height, width, nchannels, psize, stride);
}

void avepooling_forward_double(
        const double* image, double* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    avepooling_forward<double>(image, pooled, height, width, nchannels, psize,
            stride);
}

void avepooling_backward_double(
        double* image_grad, const double* pooled_grad, const int height, 
        const int width, const int nchannels, const int psize,
        const int stride) {
    avepooling_backward<double>(image_grad, pooled_grad, height,width,
            nchannels, psize, stride);
}

} // extern "C"

