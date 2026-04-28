"""Generate a small MAT input file for the FFT-DI example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io


def main() -> None:
    n = 257
    dx = 2.0e-6
    wavelength = 532e-9
    waist = 80.0e-6
    aperture_radius = 180.0e-6

    x = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    radius = np.sqrt(X**2 + Y**2)
    aperture = radius <= aperture_radius
    u0 = np.exp(-(radius / waist) ** 2) * aperture

    output = Path(__file__).with_name("sample_input.mat")
    scipy.io.savemat(output, {"u0": u0.astype(np.complex128), "lambda": wavelength, "x": x, "y": x})
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
