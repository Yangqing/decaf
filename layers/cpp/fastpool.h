#ifndef _DECAF_FASTPOOL_H
#define _DECAF_FASTPOOL_H

#define EQUAL_THRESHOLD 1e-8


extern "C" {

void maxpooling_forward_float(
        const float* image, float* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride);

void maxpooling_backward_float(
        const float* image, const float* pooled, float* image_grad,
        const float* pooled_grad, const int height, const int width,
        const int nchannels, const int psize, const int stride);

void avepooling_forward_float(
        const float* image, float* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride);

void avepooling_backward_float(
        float* image_grad, const float* pooled_grad, const int height, 
        const int width, const int nchannels, const int psize,
        const int stride);

void maxpooling_forward_double(
        const double* image, double* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride);

void maxpooling_backward_double(
        const double* image, const double* pooled, double* image_grad,
        const double* pooled_grad, const int height, const int width,
        const int nchannels, const int psize, const int stride);

void avepooling_forward_double(
        const double* image, double* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride);

void avepooling_backward_double(
        double* image_grad, const double* pooled_grad, const int height, 
        const int width, const int nchannels, const int psize,
        const int stride);

} // extern "C"

#endif // _DECAF_FASTPOOL_H
