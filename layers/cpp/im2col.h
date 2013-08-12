#ifndef _DECAF_IM2COL_H
#define _DECAF_IM2COL_H

extern "C" {

void im2col_mc_float(const float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        float* data_col);

void im2col_mc_double(const double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        double* data_col);

void col2im_mc_float(float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const float* data_col);

void col2im_mc_double(double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const double* data_col);

void im2col_sc_float(const float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        float* data_col);

void im2col_sc_double(const double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        double* data_col);

void col2im_sc_float(float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const float* data_col);

void col2im_sc_double(double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const double* data_col);

} // extern "C"

#endif // _DECAF_IM2COL_H
