import cPickle as pickle
from decaf import base
from decaf.util import translator
from decaf.util import visualize
from matplotlib import pyplot
import numpy as np
import os
import unittest

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'convnet_data')
_HAS_TEST_DATA = os.path.exists(os.path.join(_TEST_DATA_DIR, 'layers.pickle'))
_BATCH_SIZE = 32


@unittest.skipIf(not _HAS_TEST_DATA, 
                 'No cuda convnet test data found. Run'
                 ' convnet_data/get_decaf_testdata.sh to get the test data.')
class TestCudaConv(unittest.TestCase):
    """Test the mpi module
    """
    def setUp(self):
        self._layers = pickle.load(open(os.path.join(_TEST_DATA_DIR,
                                                     'layers.pickle')))
        self._data = pickle.load(open(os.path.join(_TEST_DATA_DIR,
                                                   'data',
                                                   'data_batch_5')))
        self._decaf_data = translator.imgs_cudaconv_to_decaf(
            self._data['data'][:_BATCH_SIZE], 32, 3)
        #self._decaf_labels = self._data['labels'].flatten()[:_BATCH_SIZE]
        #self._decaf_labels = self._decaf_labels.astype(np.int)
        self._output_shapes = {'data': (32, 32, 3), 'labels': -1}
        self._net = translator.translate_cuda_network(
            self._layers, self._output_shapes)
        self._net.predict(data=self._decaf_data)
        #visualize.draw_net_to_file(self._net, 'test.png')

    def testConv1(self):
        output = self._net.blobs[self._net.provides['conv1'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 32, 32, 32))
        conv_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'conv1', 'data_batch_5')))
        conv_data = translator.imgs_cudaconv_to_decaf(
            conv_data['data'][:_BATCH_SIZE], 32, 32)
        #visualize.show_channels(np.hstack((conv_data[0], output[0], conv_data[0] - output[0])))
        #pyplot.show()
        ## normalize the data before comparison
        maxval = conv_data.max()
        conv_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            conv_data.min(), conv_data.max(), np.abs(conv_data - output).max())
        np.testing.assert_array_almost_equal(conv_data, output)

    def testPool1(self):
        print self._net.layers['pool1'].spec
        output = self._net.blobs[self._net.provides['pool1'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 16, 16, 32))
        # load the data
        pool_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'pool1', 'data_batch_5')))
        pool_data = translator.imgs_cudaconv_to_decaf(
            pool_data['data'][:_BATCH_SIZE], 16, 32)
        #visualize.show_channels(np.hstack((pool_data[0], output[0], pool_data[0] - output[0])))
        #pyplot.show()
        maxval = pool_data.max()
        pool_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            pool_data.min(), pool_data.max(), np.abs(pool_data - output).max())
        np.testing.assert_array_almost_equal(pool_data, output)

    def testPool1Neuron(self):
        print self._net.layers['pool1_neuron'].spec
        output = self._net.blobs[self._net.provides['pool1_neuron'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 16, 16, 32))
        # load the data
        pool_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'pool1_neuron', 'data_batch_5')))
        pool_data = translator.imgs_cudaconv_to_decaf(
            pool_data['data'][:_BATCH_SIZE], 16, 32)
        #visualize.show_channels(np.hstack((pool_data[0], output[0], pool_data[0] - output[0])))
        #pyplot.show()
        maxval = pool_data.max()
        pool_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            pool_data.min(), pool_data.max(), np.abs(pool_data - output).max())
        np.testing.assert_array_almost_equal(pool_data, output)

    def testRnorm1(self):
        print self._net.layers['rnorm1'].spec
        output = self._net.blobs[self._net.provides['rnorm1'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 16, 16, 32))
        # load the data
        norm_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'rnorm1', 'data_batch_5')))
        norm_data = translator.imgs_cudaconv_to_decaf(
            norm_data['data'][:_BATCH_SIZE], 16, 32)
        #visualize.show_channels(np.hstack((norm_data[0], output[0], norm_data[0] - output[0])))
        #pyplot.show()
        maxval = norm_data.max()
        norm_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            norm_data.min(), norm_data.max(), np.abs(norm_data - output).max())
        np.testing.assert_array_almost_equal(norm_data, output)

    def testConv2(self):
        output = self._net.blobs[self._net.provides['conv2_neuron'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 16, 16, 64))
        conv_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'conv2', 'data_batch_5')))
        conv_data = translator.imgs_cudaconv_to_decaf(
            conv_data['data'][:_BATCH_SIZE], 16, 64)
        #visualize.show_channels(np.hstack((conv_data[0], output[0], conv_data[0] - output[0])))
        #pyplot.colorbar()
        #pyplot.show()
        ## normalize the data before comparison
        maxval = conv_data.max()
        conv_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            conv_data.min(), conv_data.max(), np.abs(conv_data - output).max())
        np.testing.assert_array_almost_equal(conv_data, output)

    def testPool2(self):
        print self._net.layers['pool2'].spec
        output = self._net.blobs[self._net.provides['pool2'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 8, 8, 64))
        # load the data
        pool_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'pool2', 'data_batch_5')))
        pool_data = translator.imgs_cudaconv_to_decaf(
            pool_data['data'][:_BATCH_SIZE], 8, 64)
        #visualize.show_channels(np.hstack((pool_data[0], output[0], pool_data[0] - output[0])))
        #pyplot.show()
        maxval = pool_data.max()
        pool_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            pool_data.min(), pool_data.max(), np.abs(pool_data - output).max())
        np.testing.assert_array_almost_equal(pool_data, output)

    def testConv3(self):
        output = self._net.blobs[self._net.provides['conv3_neuron'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 8, 8, 64))
        conv_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'conv3', 'data_batch_5')))
        conv_data = translator.imgs_cudaconv_to_decaf(
            conv_data['data'][:_BATCH_SIZE], 8, 64)
        #visualize.show_channels(np.hstack((conv_data[0], output[0], conv_data[0] - output[0])))
        #pyplot.colorbar()
        #pyplot.show()
        ## normalize the data before comparison
        maxval = conv_data.max()
        conv_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            conv_data.min(), conv_data.max(), np.abs(conv_data - output).max())
        # Due to accumulated error between CPU and GPU we relax the decimal a little bit.
        np.testing.assert_array_almost_equal(conv_data, output, 5)

    def testPool3(self):
        print self._net.layers['pool3'].spec
        output = self._net.blobs[self._net.provides['pool3'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 4, 4, 64))
        # load the data
        pool_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'pool3', 'data_batch_5')))
        pool_data = translator.imgs_cudaconv_to_decaf(
            pool_data['data'][:_BATCH_SIZE], 4, 64)
        #visualize.show_channels(np.hstack((pool_data[0], output[0], pool_data[0] - output[0])))
        #pyplot.show()
        maxval = pool_data.max()
        pool_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            pool_data.min(), pool_data.max(), np.abs(pool_data - output).max())
        np.testing.assert_array_almost_equal(pool_data, output, 6)

    def testFc64(self):
        print self._net.layers['fc64_neuron'].spec
        output = self._net.blobs[self._net.provides['fc64_neuron'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 64))
        # load the data
        fc_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'fc64_neuron', 'data_batch_5')))
        fc_data = fc_data['data'][:_BATCH_SIZE]
        #pyplot.imshow(np.hstack((fc_data, output, fc_data - output)))
        #pyplot.colorbar()
        #pyplot.show()
        maxval = fc_data.max()
        fc_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            fc_data.min(), fc_data.max(), np.abs(fc_data - output).max())
        np.testing.assert_array_almost_equal(fc_data, output)

    def testFc10(self):
        print self._net.layers['fc10'].spec
        output = self._net.blobs[self._net.provides['fc10'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 10))
        # load the data
        fc_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'fc10', 'data_batch_5')))
        fc_data = fc_data['data'][:_BATCH_SIZE]
        #pyplot.imshow(np.hstack((fc_data, output, fc_data - output)))
        #pyplot.colorbar()
        #pyplot.show()
        maxval = fc_data.max()
        fc_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            fc_data.min(), fc_data.max(), np.abs(fc_data - output).max())
        np.testing.assert_array_almost_equal(fc_data, output)

    def testProbs(self):
        print self._net.layers['probs'].spec
        output = self._net.blobs[self._net.provides['probs'][0]].data()
        self.assertEqual(output.shape, (_BATCH_SIZE, 10))
        # load the data
        prob_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, 'probs', 'data_batch_5')))
        prob_data = prob_data['data'][:_BATCH_SIZE]
        #pyplot.imshow(np.hstack((prob_data, output, prob_data - output)))
        #pyplot.colorbar()
        #pyplot.show()
        maxval = prob_data.max()
        prob_data /= maxval
        output /= maxval
        print 'data range: [%f, %f], max diff: %f' % (
            prob_data.min(), prob_data.max(), np.abs(prob_data - output).max())
        np.testing.assert_array_almost_equal(prob_data, output)

if __name__ == '__main__':
    unittest.main()

