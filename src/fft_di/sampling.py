"""Sampling checks for FFT-DI propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingReport:
    rho_max: float
    delta_rho: float
    max_dx: float
    required_n: int
    sin_theta_max: float
    nyquist_limit: float
    warnings: tuple[str, ...] = ()


def calc_min_sampling_interval(wavelength: float, rho: float, z: float) -> float:
    """Return the minimum oscillation interval estimate for the RS kernel."""
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    if rho < 0:
        raise ValueError("rho must be non-negative.")
    return math.sqrt(wavelength**2 + rho**2 + 2.0 * wavelength * math.sqrt(rho**2 + z**2)) - rho


def required_samples(physical_size: float, max_dx: float) -> int:
    """Return the minimum sample count for a physical width and max spacing."""
    if physical_size <= 0:
        raise ValueError("physical_size must be positive.")
    if max_dx <= 0:
        raise ValueError("max_dx must be positive.")
    return int(math.ceil(physical_size / max_dx)) + 1


def validate_sampling(
    *,
    dx: float,
    n_samples: int,
    physical_size: float,
    wavelength: float,
    z: float,
    theta_max_deg: float = 0.0,
    rho_mode: str = "corner",
    rho_custom: float | None = None,
    use_simpson: bool = False,
    strict: bool = True,
) -> SamplingReport:
    """Validate common FFT-DI sampling constraints.

    When strict is False, violations are returned as warnings instead of
    raising ValueError.
    """
    if use_simpson and n_samples % 2 == 0:
        raise ValueError("Simpson integration requires an odd number of samples.")
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if physical_size <= 0:
        raise ValueError("physical_size must be positive.")
    if theta_max_deg < 0:
        raise ValueError("theta_max_deg must be non-negative.")

    half_size = 0.5 * physical_size
    rho_mode = rho_mode.lower().strip()
    if rho_mode == "corner":
        rho_max = math.sqrt(2.0) * half_size
    elif rho_mode == "edge":
        rho_max = half_size
    elif rho_mode == "custom":
        if rho_custom is None or rho_custom <= 0:
            raise ValueError("rho_custom must be positive when rho_mode='custom'.")
        rho_max = float(rho_custom)
    else:
        raise ValueError("rho_mode must be one of: corner, edge, custom.")

    delta_rho = calc_min_sampling_interval(wavelength, rho_max, abs(z))
    max_dx = delta_rho / 4.0
    min_n = required_samples(physical_size, max_dx)
    warnings: list[str] = []

    if dx > max_dx:
        message = (
            f"Sampling too coarse for RS kernel: dx={dx:.4e} m, "
            f"required dx <= {max_dx:.4e} m; N >= {min_n} for this window."
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)

    sin_theta_max = math.sin(math.radians(theta_max_deg))
    nyquist_limit = wavelength / (2.0 * dx)
    if sin_theta_max >= nyquist_limit:
        message = (
            f"Nyquist condition violated: sin(theta_max)={sin_theta_max:.4f}, "
            f"lambda/(2*dx)={nyquist_limit:.4f}."
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)

    return SamplingReport(
        rho_max=rho_max,
        delta_rho=delta_rho,
        max_dx=max_dx,
        required_n=min_n,
        sin_theta_max=sin_theta_max,
        nyquist_limit=nyquist_limit,
        warnings=tuple(warnings),
    )
