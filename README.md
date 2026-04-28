# FFT-DI

FFT-DI is a compact Python implementation of the FFT-based direct integration
method for the Rayleigh-Sommerfeld diffraction integral. The project is aimed
at researchers and engineers who need a readable, modifiable baseline for
scalar diffraction propagation rather than a closed optical-design package.

This repository is the open-source version of the original Chinese project
notes, which are preserved in [README.zh-CN.md](README.zh-CN.md).

## What It Does

FFT-DI computes propagation between two parallel planes by treating the
Rayleigh-Sommerfeld diffraction integral as a linear convolution and evaluating
that convolution with FFTs. The implementation follows:

> Fabin Shen and Anbo Wang, "Fast-Fourier-transform based numerical integration
> method for the Rayleigh-Sommerfeld diffraction formula," Applied Optics 45(6),
> 1102-1110 (2006). DOI: `10.1364/AO.45.001102`.

The package is useful for:

- scalar diffraction propagation in homogeneous media;
- near-field and far-field wave-optics simulations;
- holography and diffractive optical element prototyping;
- validation of angular-spectrum or Fresnel propagation results;
- research workflows that exchange fields with MATLAB through MAT/NPZ files.

## Main Features

- Rayleigh-Sommerfeld propagation by FFT-based direct integration.
- `(2N-1, 2N-1)` zero padding for linear convolution.
- Optional Simpson-rule integration weights for odd-sized grids.
- NumPy/SciPy backend by default.
- Optional PyTorch tensor backend for GPU workflows.
- Command-line runner for plane, line-profile, and volume-scan outputs.
- Small generated example, so the repository does not need to store binary data.

## Installation

```bash
git clone https://github.com/zc345922764-hue/FFT-DI.git
cd FFT-DI
python -m pip install -e .
```

For optional PyTorch support:

```bash
python -m pip install -e ".[torch]"
```

## Quick Start

Generate a sample input field:

```bash
python examples/generate_sample.py
```

Run FFT-DI propagation:

```bash
python -m fft_di examples/config.ini
```

The output is written to `results/sample_output.npz`.

After installation, the console entry point is also available:

```bash
fft-di examples/config.ini
```

## Python API

```python
import numpy as np

from fft_di import fft_di_propagate

n = 257
dx = 2.0e-6
wavelength = 532e-9
z = 0.02

x = (np.arange(n) - (n - 1) / 2) * dx
X, Y = np.meshgrid(x, x, indexing="xy")
u0 = np.exp(-(X**2 + Y**2) / (80e-6)**2)

u1 = fft_di_propagate(u0, dx, wavelength, z, use_simpson=True)
intensity = np.abs(u1) ** 2
```

## Command-Line Config

The CLI uses an INI file. See [examples/config.ini](examples/config.ini).

```ini
[io]
input_mat = sample_input.mat
output_npz = ../results/sample_output.npz
field_key = u0
wavelength_keys = lambda,wavelength

[run]
mode = plane
backend = numpy
target_z_m = 0.02
use_simpson = true

[validation]
strict_checks = false
theta_max_deg = 0.0
rho_mode = corner
rho_custom_m = auto
```

Input MAT files should contain:

- `u0`: a square 2D real or complex field.
- `lambda` or `wavelength`: wavelength in meters.
- `x`/`y` coordinate vectors, or `dx`, or a physical size key such as `Lx`.

## Algorithm Summary

The Rayleigh-Sommerfeld impulse response used here is:

```text
g(x,y,z) = exp(i k r) / (2 pi r) * z/r * (1/r - i k)
r = sqrt(x^2 + y^2 + z^2)
k = 2 pi / wavelength
```

The discrete propagation is evaluated as:

```text
U(z) = IFFT2(FFT2(pad(U0)) * FFT2(g)) * dx * dy
```

Because FFTs naturally compute circular convolution, the field and kernel are
represented on a `(2N-1) x (2N-1)` grid before the valid `N x N` observation
window is cropped out. For odd `N`, Simpson-rule weights can be applied before
the convolution.

More implementation detail is available in [docs/algorithm.md](docs/algorithm.md).

## Scope And Limitations

This is a scalar diffraction solver. It does not model vector polarization,
material dispersion, multiple scattering, or electromagnetic boundary effects.
For high-NA systems, sharp apertures, or strongly structured fields, convergence
tests with smaller `dx`, larger windows, and independent propagation methods are
recommended.

The command-line sampling checks are practical guardrails. They are not a proof
that a simulation is physically converged.

## Repository Layout

```text
src/fft_di/          Python package
examples/            Generated sample input and config
tests/               Basic unit tests
docs/                Algorithm notes
README.zh-CN.md      Original Chinese project notes
```

This public repository intentionally excludes private IDE files, virtual
environments, generated results, binary sample data, and copyrighted paper PDFs.

## Testing

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

## Citation

If this implementation helps your work, please cite the FFT-DI method paper:

```text
Fabin Shen and Anbo Wang,
"Fast-Fourier-transform based numerical integration method for the
Rayleigh-Sommerfeld diffraction formula,"
Applied Optics 45(6), 1102-1110 (2006).
DOI: 10.1364/AO.45.001102
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
