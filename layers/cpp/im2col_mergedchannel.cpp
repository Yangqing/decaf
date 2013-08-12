// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cstring>
#include <cmath>

#include "im2col.h"

// Note: in testing the code we found that OMP is slower fore relatively small
// images (which is often the case), so we disabled OMP.

template <typename Dtype>
inline void im2col_mc(const Dtype* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        Dtype* data_col) {
    // The naive im2col_mc implementation
    int step_im = width * nchannels;
    int step_col = psize * nchannels;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;
    for (int idxh = 0; idxh < height_col; ++idxh) {
        Dtype* pointer_col = data_col + idxh * width_col * psize * step_col;
        for (int idxw = 0; idxw < width_col; ++idxw) {
            // copy image[idxh:idxh+psize, idxw:idxw+psize, :]
            int hstart = idxh * stride;
            const Dtype* pointer_im = data_im + (hstart * width + idxw * stride) * nchannels;
            for (int i = hstart; i < hstart + psize; ++i) {
                // copy image[i, idxw:idxw+psize, :]
                for (int j = 0; j < step_col; ++j) {
                    pointer_col[j] = pointer_im[j];
                }
                pointer_col += step_col;
                pointer_im += step_im;
            }
        }
    }
} // im2col_mc


template <typename Dtype>
inline void col2im_mc(Dtype* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const Dtype* data_col) {
    memset(data_im, 0, sizeof(Dtype) * height * width * nchannels);
    int step_im = width * nchannels;
    int step_col = psize * nchannels;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;
    for (int idxh = 0; idxh < height_col; ++idxh) {
        const Dtype* pointer_col = data_col + idxh * width_col * psize * step_col;
        for (int idxw = 0; idxw < width_col; ++idxw) {
            // copy image[idxh:idxh+psize, idxw:idxw+psize, :]
            int hstart = idxh * stride;
            Dtype* pointer_im = data_im + (hstart * width + idxw * stride) * nchannels;
            for (int i = hstart; i < hstart + psize; ++i) {
                // Add image[i, idxw:idxw+psize, :]
                for (int j = 0; j < step_col; ++j) {
                    pointer_im[j] += pointer_col[j];
                }
                pointer_col += step_col;
                pointer_im += step_im;
            }
        }
    }
} // im2col_mc


extern "C" {

void im2col_mc_float(const float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        float* data_col) {
    im2col_mc<float>(data_im, height, width, nchannels, psize, stride, data_col);
}

void im2col_mc_double(const double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        double* data_col) {
    im2col_mc<double>(data_im, height, width, nchannels, psize, stride, data_col);
}

void col2im_mc_float(float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const float* data_col) {
    col2im_mc<float>(data_im, height, width, nchannels, psize, stride, data_col);
}

void col2im_mc_double(double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const double* data_col) {
    col2im_mc<double>(data_im, height, width, nchannels, psize, stride, data_col);
}

} // extern "C"
