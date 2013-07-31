// The fast pooling code implemented in C
// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cstring>
#include <cmath>

#include <omp.h>

template <typename Dtype>
inline void im2col(const Dtype* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        Dtype* data_col) {
    // The naive im2col implementation
    int step_im = width * nchannels;
    int step_col = psize * nchannels;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;

#pragma omp parallel for
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
} // im2col

/* Not used. Does not provide speedup. Will delete in the future.
template <typename Dtype>
inline void im2col_per_row(const Dtype* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        Dtype* data_col) {
    // The naive im2col implementation
    int step_im = width * nchannels;
    int step_col = psize * nchannels;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;
#pragma omp parallel for
    for (int idxh = 0; idxh < height_col; ++idxh) {
        // here what we do is to have a per-row copy.
        for (int row = 0; row < psize; ++row) {
            Dtype* pointer_col = data_col + 
                idxh * width_col * psize * step_col + row * step_col;
            const Dtype* pointer_im = data_im + 
                (idxh * stride + row) * step_im;
            for (int idxw = 0; idxw < width_col; ++idxw) {
                // copy image[i, idxw:idxw+psize, :]
                for (int j = 0; j < step_col; ++j) {
                    pointer_col[j] = pointer_im[j];
                }
                pointer_col += psize * step_col;
                pointer_im += stride * nchannels;
            }
        }
    }
} // im2col_per_row
*/

template <typename Dtype>
inline void col2im(Dtype* data_im,
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
#pragma omp parallel for
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
} // im2col


extern "C" {

void im2col_float(const float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        float* data_col) {
    im2col<float>(data_im, height, width, nchannels, psize, stride, data_col);
}

void im2col_double(const double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        double* data_col) {
    im2col<double>(data_im, height, width, nchannels, psize, stride, data_col);
}

void col2im_float(float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const float* data_col) {
    col2im<float>(data_im, height, width, nchannels, psize, stride, data_col);
}

void col2im_double(double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const double* data_col) {
    col2im<double>(data_im, height, width, nchannels, psize, stride, data_col);
}

} // extern "C"
