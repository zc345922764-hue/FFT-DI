import unittest

from fft_di import validate_sampling


class SamplingTests(unittest.TestCase):
    def test_sampling_report(self):
        report = validate_sampling(
            dx=1.0e-6,
            n_samples=33,
            physical_size=32.0e-6,
            wavelength=532.0e-9,
            z=1.0e-3,
            strict=False,
            use_simpson=True,
        )
        self.assertGreater(report.max_dx, 0.0)
        self.assertGreater(report.required_n, 0)


if __name__ == "__main__":
    unittest.main()
