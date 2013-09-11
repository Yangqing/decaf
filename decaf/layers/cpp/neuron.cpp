#include <algorithm>
#ifdef DECAF_USE_MKL
#include <mkl_vml.h>
#endif // DECAF_USE_MKL
#include "neuron.h"

using std::max;

template <typename Dtype>
inline void _relu_forward(const Dtype* input, Dtype* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = max(input[i], Dtype(0));
    }
    return;
}

extern "C" {

void relu_forward(const int len, const void* input, void* output, int n) {
#ifdef DECAF_USE_MKL
    switch(len) {
    case sizeof(float):
        vsAbs(n, (const float*)input, (float*)output);
        vsAdd(n, (const float*)input, (const float*)output, (float*)output);
        break;
    case sizeof(double):
        vdAbs(n, (const double*)input, (double*)output);
        vdAdd(n, (const double*)input, (const double*)output, (double*)output);
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(len)

#else
    switch(len) {
    case sizeof(float):
        _relu_forward<float>((const float*) input, (float*) output, n);
        break;
    case sizeof(double):
        _relu_forward<double>((const double*) input, (double*) output, n);
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(len)
#endif // DECAF_USE_MKL
}

}
