from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        pass

    def testMeanNormalizeLayer(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        checker = gradcheck.GradChecker(1e-5)
        shapes = [(1,5,5,1), (1,5,5,3), (5,5), (1,5)]
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            layer = core_layers.MeanNormalizeLayer(
                name='normalize')
            result = checker.check(layer, [input_blob], [output_blob])
            print(result)
            self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
