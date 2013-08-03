from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestLossGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testSquaredLossGrad(self):
        np.random.seed(1701)
        shapes = [(4,3), (1,10), (4,3,2)]
        layer = core_layers.SquaredLossLayer(name='squared')
        checker = gradcheck.GradChecker(1e-6)
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            target_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            result = checker.check(layer, [input_blob,target_blob], [],
                                   check_indices = [0])
            print(result)
            self.assertTrue(result[0])

    def testMultinomialLogisticLossGrad(self):
        np.random.seed(1701)
        layer = core_layers.MultinomialLogisticLossLayer(name='loss')
        checker = gradcheck.GradChecker(1e-6)
        shape = (10,5)
        # check index input
        input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
        target_blob = base.Blob(shape[:1], dtype=np.int,
                                filler=fillers.RandIntFiller(high=shape[1]))
        result = checker.check(layer, [input_blob, target_blob], [],
                               check_indices = [0])
        print(result)
        self.assertTrue(result[0])
        # check full input
        target_blob = base.Blob(shape, filler=fillers.RandFiller())
        # normalize target
        target_data = target_blob.data()
        target_data /= target_data.sum(1)[:, np.newaxis]
        result = checker.check(layer, [input_blob, target_blob], [],
                               check_indices = [0])
        print(result)
        self.assertTrue(result[0])

if __name__ == '__main__':
    unittest.main()
