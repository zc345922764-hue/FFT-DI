# Algorithm Notes

This package implements the FFT-based direct integration method for the
Rayleigh-Sommerfeld diffraction integral.

## Rayleigh-Sommerfeld Kernel

For a homogeneous medium, the scalar Rayleigh-Sommerfeld propagation from an
input plane to a parallel observation plane can be written as a convolution:

```text
U(x, y, z) = integral integral U0(xi, eta) g(x - xi, y - eta, z) dxi deta
```

The impulse response used here is:

```text
g(x,y,z) = exp(i k r) / (2 pi r) * z/r * (1/r - i k)
r = sqrt(x^2 + y^2 + z^2)
k = 2 pi / wavelength
```

## FFT-DI Discretization

Direct numerical integration over an `N x N` field has high computational cost.
FFT-DI treats the Riemann sum as a 2D linear convolution:

```text
S = IFFT2(FFT2(U) * FFT2(H)) * dx * dy
```

Because FFTs compute circular convolution, the field and kernel are represented
on a `(2N-1) x (2N-1)` grid. The propagated field is then cropped from the valid
linear-convolution region.

## Simpson Weights

When `N` is odd, Simpson-rule weights can be applied before propagation:

```text
W = outer([1, 4, 2, 4, ..., 2, 4, 1],
          [1, 4, 2, 4, ..., 2, 4, 1])
scale = (dx / 3)^2
```

This improves the numerical integration order compared with the rectangular
Riemann sum.

## Sampling Checks

The helper `validate_sampling` estimates whether the Rayleigh-Sommerfeld kernel
is sampled finely enough over the requested physical window. It also checks a
basic angular Nyquist condition:

```text
sin(theta_max) < wavelength / (2 dx)
```

These checks are practical guardrails, not a substitute for convergence tests.
For high numerical aperture systems, abrupt apertures, or strongly structured
fields, repeat the calculation at smaller `dx` and larger windows.

## Reference

Fabin Shen and Anbo Wang, "Fast-Fourier-transform based numerical integration
method for the Rayleigh-Sommerfeld diffraction formula," Applied Optics 45(6),
1102-1110 (2006). DOI: `10.1364/AO.45.001102`.
