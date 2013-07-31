#include <cstring>
#include <cmath>

#include <omp.h>

template <typename Dtype>
inline void im2col_sc(const Dtype* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        Dtype* data_col) {
    int im_channel_size = height * width;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;
    // each single image channel will correspond to multiple col channels
    int col_channel_size = height_col * width_col * psize * psize;
# pragma omp parallel for
    for (int channel_id = 0; channel_id < nchannels; ++channel_id) {
        const Dtype* pointer_im = data_im + channel_id * im_channel_size;
        Dtype* pointer_col = data_col + channel_id * col_channel_size;
        // Copy the channel.
        for (int idxh = 0; idxh < height_col; ++idxh) {
            for (int idxw = 0; idxw < width_col; ++idxw) {
                const Dtype* patch_im = pointer_im + (idxh * width + idxw) * stride;
                Dtype* patch_col = pointer_col + (idxh * width_col + idxw) * psize * psize;
                for (int i = 0; i < psize; ++i) {
                    for (int j = 0; j < psize; ++j) {
                        patch_col[j] = patch_im[j];
                    }
                    patch_col += psize;
                    patch_im += width;
                }
            }
        }
    }
} // im2col_sc

template <typename Dtype>
inline void col2im_sc(Dtype* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const Dtype* data_col) {
    memset(data_im, 0, sizeof(Dtype) * height * width * nchannels);
    int im_channel_size = height * width;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;
    // each single image channel will correspond to multiple col channels
    int col_channel_size = height_col * width_col * psize * psize;
# pragma omp parallel for
    for (int channel_id = 0; channel_id < nchannels; ++channel_id) {
        Dtype* pointer_im = data_im + channel_id * im_channel_size;
        const Dtype* pointer_col = data_col + channel_id * col_channel_size;
        // Copy the channel.
        for (int idxh = 0; idxh < height_col; ++idxh) {
            for (int idxw = 0; idxw < width_col; ++idxw) {
                Dtype* patch_im = pointer_im + (idxh * width + idxw) * stride;
                const Dtype* patch_col = pointer_col + (idxh * width_col + idxw) * psize * psize;
                for (int i = 0; i < psize; ++i) {
                    for (int j = 0; j < psize; ++j) {
                        // Note: since the omp threads are going to work on separate
                        // channels, we don't need to impose locking.
                        patch_im[j] += patch_col[j];
                    }
                    patch_col += psize;
                    patch_im += width;
                }
            }
        }
    }
} // col2im_sc

extern "C" {

void im2col_sc_float(const float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        float* data_col) {
    im2col_sc<float>(data_im, height, width, nchannels, psize, stride, data_col);
}

void im2col_sc_double(const double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        double* data_col) {
    im2col_sc<double>(data_im, height, width, nchannels, psize, stride, data_col);
}

void col2im_sc_float(float* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const float* data_col) {
    col2im_sc<float>(data_im, height, width, nchannels, psize, stride, data_col);
}

void col2im_sc_double(double* data_im,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride,
        const double* data_col) {
    col2im_sc<double>(data_im, height, width, nchannels, psize, stride, data_col);
}

} // extern "C"
