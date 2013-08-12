#ifndef _DECAF_LOCAL_RESPONSE_NORMALIZATION_H
#define _DECAF_LOCAL_RESPONSE_NORMALIZATION_H

// C wrap functions.
extern "C" {

void lrn_forward_float(const float* bottom, float* top, float* scale,
        const int num_data, const int channels, const int size,
        const float alpha, const float beta);

void lrn_forward_double(const double* bottom, double* top, double* scale,
        const int num_data, const int channels, const int size,
        const double alpha, const double beta);

void lrn_backward_float(const float* bottom, const float* top,
        float* bottom_diff, const float* top_diff, const float* scale,
        const int num_data, const int channels, const int size,
        const float alpha, const float beta);

void lrn_backward_double(const double* bottom, const double* top,
        double* bottom_diff, const double* top_diff, const double* scale,
        const int num_data, const int channels, const int size,
        const double alpha, const double beta);

} // extern "C"

#endif // _DECAF_LOCAL_RESPONSE_NORMALIZATION_H
