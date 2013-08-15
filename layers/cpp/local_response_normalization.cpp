#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include "local_response_normalization.h"

// Takes a bottom input of size (num_data * channels), computes the local
// response normalized output, saving the intermediate scale values in scale.
// See Alex Krizhevsky's cudaconv documentation for more details.
template <typename Dtype>
inline void _lrn_forward(const Dtype* bottom, Dtype* top, Dtype* scale,
        const int num_data, const int channels, const int size,
        const Dtype alpha, const Dtype beta) {
    // Iterates over the data.
    int padded_channels = channels + size - 1;
    int pre_pad = (size - 1) / 2;
#pragma omp parallel
    {
    Dtype * padded_square = new Dtype[padded_channels];
    memset(padded_square, 0, sizeof(Dtype) * padded_channels);
#pragma omp for
    for (int data_id = 0; data_id < num_data; ++data_id) {
        const Dtype* bottom_datum = bottom + data_id * channels;
        Dtype* top_datum = top + data_id * channels;
        Dtype* scale_datum = scale + data_id * channels;
        // first, compute x_i^2
        for (int i = 0; i < channels; ++i) {
            padded_square[i+pre_pad] = bottom_datum[i] * bottom_datum[i] 
                    * alpha / size; 
        }
        // Now, compute the running scale.
        Dtype accum_scale = 0.;
        for (int i = 0; i < size - 1; ++i) {
            accum_scale += padded_square[i];
        }
        for (int i = 0; i < channels; ++i) {
            accum_scale += padded_square[i + size - 1];
            scale_datum[i] = 1. + accum_scale;
            accum_scale -= padded_square[i];
            top_datum[i] = bottom_datum[i] * pow(scale_datum[i], -beta);
        }
    }
    delete[] padded_square;
    } // pragma omp parallel
}


template <typename Dtype>
inline void _lrn_backward(const Dtype* bottom, const Dtype* top, Dtype* bottom_diff,
        const Dtype* top_diff, const Dtype* scale, const int num_data,
        const int channels, const int size, const Dtype alpha,
        const Dtype beta) {
    int padded_channels = channels + size - 1;
    int pre_pad = size - (size + 1) / 2;
#pragma omp parallel
    {
    Dtype * padded_ratio = new Dtype[padded_channels];
    memset(padded_ratio, 0, sizeof(Dtype) * padded_channels);
    // the ratio 2*alpha*beta/size
    Dtype cache_ratio = 2. * alpha * beta / size;
#pragma omp for
    for (int data_id = 0; data_id < num_data; ++data_id) {
        const Dtype* bottom_datum = bottom + data_id * channels;
        const Dtype* top_datum = top + data_id * channels;
        const Dtype* top_diff_datum = top_diff + data_id * channels;
        const Dtype* scale_datum = scale + data_id * channels;
        Dtype* bottom_diff_datum = bottom_diff + data_id * channels;
        // first, compute diff_i * y_i / s_i
        for (int i = 0; i < channels; ++i) {
            padded_ratio[i + pre_pad] = top_diff_datum[i] * top_datum[i] 
                / scale_datum[i];
        }
        Dtype accum_ratio = 0.;
        for (int i = 0; i < size - 1; ++i) {
            accum_ratio += padded_ratio[i];
        }
        for (int i = 0; i < channels; ++i) {
            accum_ratio += padded_ratio[i + size - 1];
            bottom_diff_datum[i] = 
                top_diff_datum[i] * pow(scale_datum[i], -beta) -
                cache_ratio * bottom_datum[i] * accum_ratio;
            accum_ratio -= padded_ratio[i];
        }
    }
    delete[] padded_ratio;
    } // pragma omp parallel
}

extern "C" {

void lrn_forward(const int len, const void* bottom, void* top, void* scale,
        const int num_data, const int channels, const int size,
        const double alpha, const double beta) {
    switch(len) {
    case sizeof(float):
        _lrn_forward<float>((const float*)bottom, (float*)top, (float*)scale,
                num_data, channels, size, (float)alpha, (float)beta);
        break;
    case sizeof(double):
        _lrn_forward<double>((const double*)bottom, (double*)top, (double*)scale,
                num_data, channels, size, alpha, beta);
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(size)
}

void lrn_backward(const int len, const void* bottom, const void* top,
        void* bottom_diff, const void* top_diff, const void* scale,
        const int num_data, const int channels, const int size,
        const double alpha, const double beta) {
    switch(len) {
    case sizeof(float):
        _lrn_backward<float>((const float*)bottom, (const float*)top,
                (float*)bottom_diff, (const float*)top_diff, 
                (const float*)scale, num_data, channels, size, (float)alpha,
                (float)beta);
        break;
    case sizeof(double):
        _lrn_backward<double>((const double*)bottom, (const double*)top,
                (double*)bottom_diff, (const double*)top_diff, 
                (const double*)scale, num_data, channels, size, alpha, beta);
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(size)
}

} // extern "C"
