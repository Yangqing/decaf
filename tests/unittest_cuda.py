from decaf import cuda
import logging
import numpy as np
import numpy.testing as npt
import unittest


class TestBlob(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.skipIf(not cuda.has_cuda,
                     'No cuda runtime on the test machine.')
    def testCudaInit(self):
        self.assertTrue(cuda.init_cuda(), 0)
