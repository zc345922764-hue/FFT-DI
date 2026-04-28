import unittest

import numpy as np

from fft_di import build_rs_kernel_fft_di, fft_di_propagate, simpson_weights


class PropagatorTests(unittest.TestCase):
    def test_kernel_shape(self):
        kernel = build_rs_kernel_fft_di(17, 1.0e-6, 532.0e-9, 1.0e-3)
        self.assertEqual(kernel.shape, (33, 33))

    def test_propagation_shape(self):
        field = np.ones((17, 17), dtype=np.complex128)
        out = fft_di_propagate(field, 1.0e-6, 532.0e-9, 1.0e-3, use_simpson=True)
        self.assertEqual(out.shape, field.shape)
        self.assertTrue(np.iscomplexobj(out))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_simpson_requires_odd_samples(self):
        self.assertIsNone(simpson_weights(16))
        with self.assertRaises(ValueError):
            fft_di_propagate(np.ones((16, 16)), 1.0e-6, 532.0e-9, 1.0e-3, use_simpson=True)


if __name__ == "__main__":
    unittest.main()
