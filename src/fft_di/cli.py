"""Command line interface for FFT-DI."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import numpy as np

from .io import load_mat_field
from .propagator import fft_di_propagate
from .sampling import validate_sampling


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _get_bool(config: configparser.ConfigParser, section: str, key: str, fallback: bool) -> bool:
    if config.has_option(section, key):
        return config.getboolean(section, key)
    return fallback


def _get_optional_float(config: configparser.ConfigParser, section: str, key: str) -> float | None:
    if not config.has_option(section, key):
        return None
    value = config.get(section, key).strip()
    if value.lower() in {"", "auto", "none", "null"}:
        return None
    return float(value)


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_backend(field: np.ndarray, backend: str, device: str):
    backend = backend.lower().strip()
    if backend == "numpy":
        return np.asarray(field)
    if backend != "torch":
        raise ValueError("backend must be 'numpy' or 'torch'.")
    import torch  # type: ignore

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.from_numpy(np.asarray(field))
    if not torch.is_complex(tensor):
        tensor = tensor.to(torch.complex128)
    return tensor.to(device)


def run_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent
    config = configparser.ConfigParser()
    read_files = config.read(config_path, encoding="utf-8")
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    input_mat = _resolve(base_dir, config.get("io", "input_mat"))
    output_npz = _resolve(base_dir, config.get("io", "output_npz", fallback="results/output_fft_di.npz"))
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    field_data = load_mat_field(
        input_mat,
        field_key=config.get("io", "field_key", fallback="u0"),
        wavelength_keys=config.get("io", "wavelength_keys", fallback="lambda,wavelength"),
        x_key=config.get("io", "x_key", fallback="x"),
        y_key=config.get("io", "y_key", fallback="y"),
    )

    mode = config.get("run", "mode", fallback="plane").lower().strip()
    if mode not in {"plane", "line", "volume"}:
        raise ValueError("run.mode must be one of: plane, line, volume.")
    backend = config.get("run", "backend", fallback="numpy")
    device = config.get("run", "device", fallback="auto")
    use_simpson = _get_bool(config, "run", "use_simpson", True)
    strict = _get_bool(config, "validation", "strict_checks", True)
    theta_max_deg = config.getfloat("validation", "theta_max_deg", fallback=0.0)
    rho_mode = config.get("validation", "rho_mode", fallback="corner")
    rho_custom = _get_optional_float(config, "validation", "rho_custom_m")

    if mode == "volume":
        z_start = config.getfloat("run", "z_start_m")
        z_end = config.getfloat("run", "z_end_m")
        z_steps = config.getint("run", "z_steps")
        z_values = np.linspace(z_start, z_end, z_steps)
        z_for_checks = float(np.min(np.abs(z_values)))
    else:
        target_z = config.getfloat("run", "target_z_m")
        z_values = np.array([target_z], dtype=float)
        z_for_checks = float(target_z)

    report = validate_sampling(
        dx=field_data.dx,
        n_samples=int(field_data.field.shape[0]),
        physical_size=field_data.physical_size,
        wavelength=field_data.wavelength,
        z=z_for_checks,
        theta_max_deg=theta_max_deg,
        rho_mode=rho_mode,
        rho_custom=rho_custom,
        use_simpson=use_simpson,
        strict=strict,
    )
    for warning in report.warnings:
        print(f"Warning: {warning}")

    field = _prepare_backend(field_data.field, backend, device)
    print("Input data loaded:")
    print(f"  field: {field_data.field.shape}")
    print(f"  wavelength: {field_data.wavelength:.4e} m")
    print(f"  dx: {field_data.dx:.4e} m")
    print(f"  physical size: {field_data.physical_size:.4e} m")
    print(f"  backend: {backend}")

    if mode == "volume":
        volume = []
        for z in z_values:
            print(f"Propagating z={z:.4e} m")
            volume.append(_as_numpy(fft_di_propagate(field, field_data.dx, field_data.wavelength, float(z), use_simpson=use_simpson)))
        np.savez_compressed(output_npz, volume=np.stack(volume, axis=0), z_vec=z_values, dx=field_data.dx, wavelength=field_data.wavelength)
    else:
        z = float(z_values[0])
        print(f"Propagating z={z:.4e} m")
        result = _as_numpy(fft_di_propagate(field, field_data.dx, field_data.wavelength, z, use_simpson=use_simpson))
        if mode == "line":
            line = result[result.shape[0] // 2, :]
            np.savez_compressed(output_npz, line_profile=line, dx=field_data.dx, z=z, wavelength=field_data.wavelength)
        else:
            np.savez_compressed(output_npz, u_out=result, dx=field_data.dx, z=z, wavelength=field_data.wavelength)

    print(f"Saved output to: {output_npz}")
    return output_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rayleigh-Sommerfeld FFT-DI propagation.")
    parser.add_argument("config", type=Path, help="Path to an INI config file.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_from_config(args.config)
