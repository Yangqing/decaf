from decaf import base
from decaf.layers import fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest

class TestRegLayersGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testRegLayersGrad(self):
        np.random.seed(1701)
        output_blobs = [base.Blob()]
        checker = gradcheck.GradChecker(1e-5)
        shapes = [(5,4), (5,1), (1,5), (1,5,5), (1,5,5,3), (1,5,5,1)]
        all_classes = [regularization.L2RegularizeLayer,
                       regularization.L1RegularizeLayer,
                       regularization.AutoencoderRegularizeLayer]
        all_fillers = [fillers.GaussianRandFiller(),
                       fillers.GaussianRandFiller(),
                       fillers.RandFiller(min=0.05, max=0.95)]
        args = [{},
                {},
                {'ratio': 0.05}]

        for reg_layer_class, filler, arg in zip(all_classes, all_fillers, args):
            for shape in shapes:
                input_blob = base.Blob(shape, filler=filler)
                layer = reg_layer_class(name='reg', weight=0.1, **arg)
                result = checker.check(layer, [input_blob], output_blobs)
                print(result)
                self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
