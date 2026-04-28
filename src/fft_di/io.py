"""Input and output helpers for FFT-DI examples and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io


@dataclass(frozen=True)
class FieldData:
    field: np.ndarray
    wavelength: float
    dx: float
    physical_size: float
    x: np.ndarray | None = None
    y: np.ndarray | None = None


def _split_keys(keys: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(keys, str):
        return tuple(key.strip() for key in keys.split(",") if key.strip())
    return tuple(keys)


def _get_scalar(mat_data: dict, keys: str | Iterable[str]) -> float | None:
    for key in _split_keys(keys):
        if key in mat_data:
            value = np.squeeze(mat_data[key])
            if np.size(value) == 1:
                return float(value)
    return None


def _validate_axis_vector(vector: np.ndarray, name: str) -> tuple[int, float, float]:
    vector = np.ravel(vector)
    if vector.ndim != 1 or vector.size < 3:
        raise ValueError(f"{name} must be a 1D vector with at least 3 samples.")
    diffs = np.diff(vector)
    if not np.all(diffs > 0):
        raise ValueError(f"{name} must be strictly increasing.")
    dx = float(np.mean(diffs))
    if not np.allclose(diffs, dx, rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced.")
    return int(vector.size), dx, float(vector[-1] - vector[0])


def load_mat_field(
    path: str | Path,
    *,
    field_key: str = "u0",
    wavelength_keys: str | Iterable[str] = ("lambda", "wavelength"),
    x_key: str = "x",
    y_key: str = "y",
    dx_keys: str | Iterable[str] = ("dx", "dX", "DX"),
    size_keys: str | Iterable[str] = ("Lx", "LX", "size_x", "x_size"),
) -> FieldData:
    """Load a square 2D complex field and sampling metadata from a MAT file."""
    path = Path(path)
    mat_data = scipy.io.loadmat(path)
    if field_key not in mat_data:
        raise KeyError(f"MAT file does not contain field key: {field_key}")
    field = np.squeeze(mat_data[field_key])
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError(f"field must be square 2D data; got shape {field.shape}.")

    wavelength = _get_scalar(mat_data, wavelength_keys)
    if wavelength is None:
        raise KeyError(f"MAT file does not contain any wavelength key: {_split_keys(wavelength_keys)}")
    n_samples = int(field.shape[0])

    x_vec = mat_data.get(x_key)
    y_vec = mat_data.get(y_key)
    dx: float | None = None
    physical_size: float | None = None
    x_out: np.ndarray | None = None
    y_out: np.ndarray | None = None

    if x_vec is not None:
        n_x, dx_x, size_x = _validate_axis_vector(x_vec, x_key)
        if n_x != n_samples:
            raise ValueError(f"Sample mismatch: len({x_key})={n_x}, field is {n_samples}x{n_samples}.")
        dx = dx_x
        physical_size = size_x
        x_out = np.ravel(x_vec)

    if y_vec is not None:
        n_y, dy_y, size_y = _validate_axis_vector(y_vec, y_key)
        if n_y != n_samples:
            raise ValueError(f"Sample mismatch: len({y_key})={n_y}, field is {n_samples}x{n_samples}.")
        if dx is None:
            dx = dy_y
            physical_size = size_y
        elif not np.isclose(dx, dy_y, rtol=1e-6, atol=1e-12):
            raise ValueError(f"dx != dy (dx={dx:.4e}, dy={dy_y:.4e}).")
        if physical_size is not None and not np.isclose(physical_size, size_y, rtol=1e-6, atol=1e-12):
            raise ValueError("Physical size differs between x and y axes.")
        y_out = np.ravel(y_vec)

    if dx is None:
        dx = _get_scalar(mat_data, dx_keys)
        if dx is not None:
            physical_size = dx * (n_samples - 1)
        else:
            physical_size = _get_scalar(mat_data, size_keys)
            if physical_size is not None:
                dx = physical_size / (n_samples - 1)

    if dx is None or physical_size is None:
        raise ValueError("Missing sampling info. Provide x/y vectors or scalar dx/Lx in the MAT file.")

    return FieldData(
        field=field,
        wavelength=float(wavelength),
        dx=float(dx),
        physical_size=float(physical_size),
        x=x_out,
        y=y_out,
    )
