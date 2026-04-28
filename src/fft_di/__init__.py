"""FFT-DI tools for Rayleigh-Sommerfeld diffraction propagation."""

from .propagator import (
    build_rs_kernel_fft_di,
    fft_di_axis,
    fft_di_pad,
    fft_di_propagate,
    fft_di_propagate_with_kernel,
    make_axis,
    simpson_weights,
)
from .sampling import (
    SamplingReport,
    calc_min_sampling_interval,
    required_samples,
    validate_sampling,
)

__all__ = [
    "SamplingReport",
    "build_rs_kernel_fft_di",
    "calc_min_sampling_interval",
    "fft_di_axis",
    "fft_di_pad",
    "fft_di_propagate",
    "fft_di_propagate_with_kernel",
    "make_axis",
    "required_samples",
    "simpson_weights",
    "validate_sampling",
]

__version__ = "0.1.0"
