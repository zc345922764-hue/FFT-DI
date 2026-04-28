"""Numerical propagation by FFT-based direct integration.

The implementation follows the Rayleigh-Sommerfeld FFT-DI formulation from
Shen and Wang, Applied Optics 45(6), 1102-1110 (2006).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

Backend = Literal["numpy", "torch"]

_TORCH: Any | None = None
_TORCH_CHECKED = False


def _get_torch() -> Any | None:
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        try:
            import torch  # type: ignore
        except ImportError:
            _TORCH = None
        else:
            _TORCH = torch
        _TORCH_CHECKED = True
    return _TORCH


def _require_torch() -> Any:
    torch = _get_torch()
    if torch is None:
        raise ImportError("PyTorch is required for backend='torch'. Install with: pip install 'fft-di[torch]'")
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _get_torch()
    return torch is not None and isinstance(value, torch.Tensor)


def _backend_from_array(value: Any) -> Backend:
    return "torch" if _is_torch_tensor(value) else "numpy"


def _numpy_real_dtype(dtype: Any | None) -> np.dtype:
    if dtype is None:
        return np.dtype(np.float64)
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.complex64):
        return np.dtype(np.float32)
    if dtype.kind == "c":
        return np.dtype(np.float64)
    if dtype.kind == "f":
        return dtype
    return np.dtype(np.float64)


def _torch_complex_dtype(real_dtype: Any) -> Any:
    torch = _require_torch()
    if real_dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return torch.complex64
    return torch.complex128


def make_axis(
    n: int,
    dx: float,
    *,
    backend: Backend = "numpy",
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Return a centered, uniformly spaced coordinate axis."""
    if n < 1:
        raise ValueError("n must be positive.")
    half = dx * (n - 1) / 2.0
    if backend == "torch":
        torch = _require_torch()
        return torch.linspace(-half, half, n, device=device, dtype=dtype or torch.float64)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'.")
    return np.linspace(-half, half, n, dtype=_numpy_real_dtype(dtype))


def fft_di_axis(axis: Any) -> Any:
    """Build the coordinate-difference axis used by the FFT-DI kernel."""
    n = int(axis.shape[0])
    if n < 1:
        raise ValueError("axis must contain at least one sample.")
    if _is_torch_tensor(axis):
        torch = _require_torch()
        axis_rev = torch.flip(axis[1:], dims=[0])
        out = torch.empty(2 * n - 1, device=axis.device, dtype=axis.dtype)
    else:
        axis = np.asarray(axis)
        axis_rev = np.flip(axis[1:])
        out = np.empty(2 * n - 1, dtype=axis.dtype)
    out[: n - 1] = axis[0] - axis_rev
    out[n - 1 :] = axis - axis[0]
    return out


def simpson_weights(
    n: int,
    *,
    backend: Backend = "numpy",
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any | None:
    """Return the 2D Simpson weight matrix, or None when n is even."""
    if n % 2 == 0:
        return None
    if backend == "torch":
        torch = _require_torch()
        w = torch.ones(n, device=device, dtype=dtype or torch.float64)
    elif backend == "numpy":
        w = np.ones(n, dtype=_numpy_real_dtype(dtype))
    else:
        raise ValueError("backend must be 'numpy' or 'torch'.")
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w[:, None] * w[None, :]


def build_rs_kernel_fft_di(
    n: int,
    dx: float,
    wavelength: float,
    z: float,
    *,
    backend: Backend = "numpy",
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Return FFT2 of the Rayleigh-Sommerfeld impulse response."""
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if z == 0:
        raise ValueError("z must be nonzero for Rayleigh-Sommerfeld propagation.")

    if backend == "torch":
        torch = _require_torch()
        real_dtype = dtype or torch.float64
        axis = make_axis(n, dx, backend="torch", device=device, dtype=real_dtype)
        x = fft_di_axis(axis)
        y = fft_di_axis(axis)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
        r = torch.sqrt(x_grid**2 + y_grid**2 + z**2)
        k0 = 2.0 * math.pi / wavelength
        g = (1.0 / (2.0 * math.pi)) * torch.exp(1j * k0 * r) / r
        g = g * (z / r) * (1.0 / r - 1j * k0)
        return torch.fft.fft2(g)

    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'.")
    real_dtype = _numpy_real_dtype(dtype)
    axis = make_axis(n, dx, backend="numpy", dtype=real_dtype)
    x = fft_di_axis(axis)
    y = fft_di_axis(axis)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(x_grid**2 + y_grid**2 + z**2)
    k0 = 2.0 * math.pi / wavelength
    g = (1.0 / (2.0 * math.pi)) * np.exp(1j * k0 * r) / r
    g = g * (z / r) * (1.0 / r - 1j * k0)
    return np.fft.fft2(g)


def fft_di_pad(field: Any) -> Any:
    """Zero-pad a square field to (2N-1, 2N-1)."""
    n = int(field.shape[-1])
    if _is_torch_tensor(field):
        import torch.nn.functional as functional  # type: ignore

        return functional.pad(field, (0, n - 1, 0, n - 1))
    return np.pad(np.asarray(field), ((0, n - 1), (0, n - 1)), mode="constant")


def _ensure_complex(field: Any) -> Any:
    if _is_torch_tensor(field):
        torch = _require_torch()
        if torch.is_complex(field):
            return field
        return field.to(_torch_complex_dtype(field.dtype))
    field = np.asarray(field)
    if np.iscomplexobj(field):
        return field
    real_dtype = _numpy_real_dtype(field.dtype)
    complex_dtype = np.complex64 if real_dtype == np.dtype(np.float32) else np.complex128
    return field.astype(complex_dtype, copy=False)


def fft_di_propagate_with_kernel(
    field: Any,
    kernel_fft: Any,
    dx: float,
    *,
    use_simpson: bool = False,
) -> Any:
    """Propagate a square 2D field using a precomputed FFT-DI kernel."""
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("field must be a square 2D array.")
    n = int(field.shape[-1])
    expected = 2 * n - 1
    if kernel_fft.shape[-2] != expected or kernel_fft.shape[-1] != expected:
        raise ValueError("kernel_fft must have shape (2N-1, 2N-1) for N=field.shape[-1].")

    backend = _backend_from_array(field)
    field = _ensure_complex(field)

    if use_simpson:
        if backend == "torch":
            weights = simpson_weights(n, backend="torch", device=field.device, dtype=field.real.dtype)
        else:
            weights = simpson_weights(n, backend="numpy", dtype=field.real.dtype)
        if weights is None:
            raise ValueError("Simpson weights require an odd number of samples.")
        field = field * weights
        scale = (dx / 3.0) ** 2
    else:
        scale = dx**2

    padded = fft_di_pad(field)
    if backend == "torch":
        torch = _require_torch()
        spectrum = torch.fft.fft2(padded) * kernel_fft
        result = torch.fft.ifft2(spectrum) * scale
    else:
        result = np.fft.ifft2(np.fft.fft2(padded) * kernel_fft) * scale
    return result[n - 1 : n - 1 + n, n - 1 : n - 1 + n]


def fft_di_propagate(
    field: Any,
    dx: float,
    wavelength: float,
    z: float,
    *,
    use_simpson: bool = False,
) -> Any:
    """Propagate a square 2D field by Rayleigh-Sommerfeld FFT-DI."""
    field = _ensure_complex(field)
    backend = _backend_from_array(field)
    if backend == "torch":
        kernel_fft = build_rs_kernel_fft_di(
            int(field.shape[-1]),
            dx,
            wavelength,
            z,
            backend="torch",
            device=field.device,
            dtype=field.real.dtype,
        )
    else:
        kernel_fft = build_rs_kernel_fft_di(
            int(field.shape[-1]),
            dx,
            wavelength,
            z,
            backend="numpy",
            dtype=field.real.dtype,
        )
    return fft_di_propagate_with_kernel(field, kernel_fft, dx, use_simpson=use_simpson)
