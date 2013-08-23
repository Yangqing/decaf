import cPickle as pickle
from decaf import puff
import logging
import numpy as np
import numpy.testing as npt
import tempfile
import unittest


class TestPuff(unittest.TestCase):
    def setUp(self):
        pass

    def testPuffVectorForm(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4)
        puff.write_puff(data, fname)
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        npt.assert_array_almost_equal(puff_recovered.read_all(), data)
        self.assertTrue(puff_recovered.read_all().shape, (4,))
        puff_recovered.seek(0)
        npt.assert_array_almost_equal(puff_recovered.read(2), data[:2])
        self.assertTrue(puff_recovered.read(2).shape, (2,))

    def testPuffSingleWrite(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4,3)
        puff.write_puff(data, fname)
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        npt.assert_array_almost_equal(puff_recovered.read_all(), data)

    def testPuffMultipleWrites(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4,3)
        writer = puff.PuffStreamedWriter(fname)
        writer.write_batch(data)
        writer.write_batch(data)
        writer.write_single(data[0])
        writer.finish()
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        data_recovered = puff_recovered.read_all()
        npt.assert_array_almost_equal(data_recovered[:4], data)
        npt.assert_array_almost_equal(data_recovered[4:8], data)
        npt.assert_array_almost_equal(data_recovered[8], data[0])

    def testPuffReadBoundary(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4,3)
        puff.write_puff(data, fname)
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        npt.assert_array_almost_equal(puff_recovered.read(3), data[:3])
        npt.assert_array_almost_equal(puff_recovered.read(3),
                                      data[np.array([3,0,1], dtype=int)])
        # test seeking
        puff_recovered.seek(1)
        npt.assert_array_almost_equal(puff_recovered.read(2), data[1:3])


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
