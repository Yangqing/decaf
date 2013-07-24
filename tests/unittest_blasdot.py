from decaf.util import blasdot
import numpy as np
import unittest

class TestBlasdot(unittest.TestCase):
    """Test the blasdot module
    """
    def setUp(self):
        self.test_matrices = [
            (np.random.rand(2,3), 
             np.random.rand(3,4)),
            (np.array(np.random.rand(4,5), order='f'),
             np.array(np.random.rand(5,2))),
            (np.array(np.random.rand(3,5)),
             np.array(np.random.rand(5,4), order='f')),
            (np.array(np.random.rand(5,3), order='f'),
             np.array(np.random.rand(3,6), order='f'))]
        self.test_matrices += \
            [(b.T, a.T) for a,b in self.test_matrices]
        self.test_matrices += \
            [(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))
                for a,b in self.test_matrices]

    def testgemm(self):
        for A, B in self.test_matrices:
            result = blasdot.gemm_c_contiguous(1., A, B)
            result_ref = np.dot(A,B)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)

    def testgemm_scale(self):
        for A, B in self.test_matrices:
            result = blasdot.gemm_c_contiguous(2., A, B)
            result_ref = np.dot(A,B) * 2.
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)

    def testgemm_with_out(self):
        for A, B in self.test_matrices:
            result_ref = np.dot(A,B)
            result = np.empty(result_ref.shape, dtype = A.dtype)
            blasdot.gemm_c_contiguous(1., A, B, out=result)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)

    def testgemm_scale_with_out(self):
        for A, B in self.test_matrices:
            result_ref = np.dot(A,B) * 2.
            result = np.empty(result_ref.shape, dtype = A.dtype)
            blasdot.gemm_c_contiguous(2., A, B, out=result)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)

    def testdot(self):
        for A, B in self.test_matrices:
            result = blasdot.dot(A, B)
            result_ref = np.dot(A,B)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)
    
    def testdot_with_out(self):
        for A, B in self.test_matrices:
            result_ref = np.dot(A,B)
            result = np.empty(result_ref.shape, dtype = A.dtype)
            blasdot.dot(A, B, out = result)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)
    
        
if __name__ == '__main__':
    unittest.main()

