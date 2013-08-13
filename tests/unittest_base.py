from decaf import base
import logging
import numpy as np
import numpy.testing as npt
import tempfile
import unittest


class TestBlob(unittest.TestCase):
    def setUp(self):
        pass

    def testBlobInit(self):
        """testBlobInit checks if blobs can be successfully initialized."""
        blob = base.Blob()
        self.assertFalse(blob.has_data())
        self.assertFalse(blob.has_diff())
        blob = base.Blob((1,1))
        self.assertTrue(blob.has_data())
        self.assertFalse(blob.has_diff())
        self.assertEqual(blob.data().shape, (1,1))
    
    def testBlobUpdate(self):
        """testBlobUpdate checks if blob update() succeeds."""
        blob = base.Blob((4,3))
        diff = blob.init_diff()
        diff[:] = 1.
        blob.update()
        npt.assert_array_almost_equal(blob.data(), - blob.diff())

    def testBlobSwap(self):
        blob_a = base.Blob((4,3))
        blob_b = base.Blob((4,3))
        blob_a.data().flat = 1.
        blob_b.data().flat = 2.
        blob_a.swap_data(blob_b)
        npt.assert_array_almost_equal(blob_a.data(), 2.)
        npt.assert_array_almost_equal(blob_b.data(), 1.)

    def testUseBlob(self):
        """testUseBlob checks if simple blob usages work."""
        blob_a = base.Blob((4,3))
        blob_b = base.Blob((3,4))
        output = np.dot(blob_a.data(), blob_b.data())
        self.assertEqual(output.shape, (4,4))
        blob_c = base.Blob((4,4))
        output = np.dot(blob_a.data().T, blob_c.data())
        self.assertEqual(output.shape, (3,4))


class TestNet(unittest.TestCase):
    def setUp(self):
        pass

    def testSplit(self):
        """testSplit tests if a net is able to insert split layers correctly.
        """
        decaf_net = base.Net()
        decaf_net.add_layer(base.Layer(name='a'), provides='data')
        decaf_net.add_layer(base.Layer(name='b'), needs='data')
        decaf_net.add_layer(base.Layer(name='c'), needs='data')
        decaf_net.finish()
        self.assertEqual(len(decaf_net.layers), 4)
        self.assertEqual(len(decaf_net.blobs), 3)
        self.assertTrue(any(isinstance(layer, base.SplitLayer)
                             for layer in decaf_net.layers.values()))
        from decaf.util import visualize
        visualize.draw_net_to_file(decaf_net, tempfile.mktemp('.png'))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
