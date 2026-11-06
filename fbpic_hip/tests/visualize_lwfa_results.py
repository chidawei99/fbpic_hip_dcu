#!/usr/bin/env python3
"""
Full post-processing and visualization for the FBPIC LWFA script `lwfa_script2.py`.

Typical usage, after the simulation has produced `diags/hdf5`:

    python visualize_lwfa_results.py \
        --diag-dir diags/hdf5 \
        --out-dir lwfa_viz \
        --input-script lwfa_script2.py \
        --species electrons \
        --frame-stride 2 \
        --summary-stride 1 \
        --animate

Outputs
-------
- quicklook_frames/*.png: per-iteration 2x2 overview
- particle_plots/*.png: energy spectrum, phase space, transverse distribution
- lineouts/*.png: on-axis wake/field lineouts
- summary.csv: time evolution of beam and field metrics
- summary_plots/*.png: evolution plots
- available_contents.txt: detected openPMD fields/species/components
- quicklook.gif: optional animation when --animate is used

Dependencies
------------
    pip install openpmd-viewer matplotlib numpy scipy imageio

Notes
-----
This script assumes the default FBPIC openPMD output directory, usually `diags/hdf5`.
It tries several field-coordinate combinations because FBPIC theta-mode datasets can expose
fields as cylindrical coordinates (`r`, `t`, `z`) or reconstructed Cartesian slices (`x`, `y`, `z`)
depending on openPMD-viewer version and backend.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.constants import c, e as elementary_charge, m_e

MEC2_MEV = 0.51099895069


@dataclass
class FieldRequest:
    field: str
    coord: Optional[str]
    label: str
    kwargs: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FBPIC LWFA openPMD diagnostics: fields, particles, spectra, lineouts, evolution, animation."
    )
    parser.add_argument("--diag-dir", default="diags/hdf5", help="openPMD HDF5 diagnostic directory, usually diags/hdf5")
    parser.add_argument("--out-dir", default="lwfa_viz", help="output directory for all plots and CSV files")
    parser.add_argument("--input-script", default=None, help="optional path to lwfa_script2.py; used to plot nominal plasma profile and infer lambda0")
    parser.add_argument("--species", default="electrons", help="particle species name in the openPMD file")
    parser.add_argument("--iterations", type=int, nargs="*", default=None, help="specific iterations to plot; default: all")
    parser.add_argument("--frame-stride", type=int, default=1, help="plot every Nth available iteration")
    parser.add_argument("--summary-stride", type=int, default=1, help="compute summary metrics every Nth available iteration")
    parser.add_argument("--plot-max-particles", type=int, default=200_000, help="maximum particles used in scatter/histogram plots")
    parser.add_argument("--phase-bins", type=int, default=250, help="number of bins in phase-space and spectrum plots")
    parser.add_argument("--bunch-threshold-MeV", type=float, default=5.0, help="energy threshold used for bunch statistics")
    parser.add_argument(
        "--charge-thresholds-MeV",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 50.0, 100.0],
        help="energy thresholds for accelerated charge curves",
    )
    parser.add_argument("--animate", action="store_true", help="make quicklook.gif from quicklook frames")
    parser.add_argument("--fps", type=float, default=6.0, help="animation frame rate")
    parser.add_argument("--skip-particle-plots", action="store_true", help="only plot field quicklooks and summary fields")
    parser.add_argument("--skip-summary", action="store_true", help="skip evolution summary metrics")
    parser.add_argument("--dpi", type=int, default=150, help="figure DPI")
    parser.add_argument("--random-seed", type=int, default=42, help="random seed used for particle downsampling in plots")
    return parser.parse_args()


def import_openpmd_viewer():
    try:
        from openpmd_viewer import OpenPMDTimeSeries  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "Could not import openPMD-viewer. Install it in the same environment as FBPIC:\n"
            "    pip install openpmd-viewer\n"
            f"Original error: {exc}"
        )
    return OpenPMDTimeSeries


def mkdirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "frames": out_dir / "quicklook_frames",
        "particles": out_dir / "particle_plots",
        "lineouts": out_dir / "lineouts",
        "summary": out_dir / "summary_plots",
        "setup": out_dir / "setup_plots",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def load_input_script(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    script = Path(path)
    if not script.exists():
        print(f"[warn] input script not found: {script}")
        return None
    try:
        spec = importlib.util.spec_from_file_location("lwfa_input_script", script)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not create module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # safe for this FBPIC script because simulation is guarded by if __name__ == '__main__'
        return module
    except Exception as exc:
        print(f"[warn] could not import input script {script}: {exc}")
        return None


def write_available_contents(ts: Any, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("openPMD diagnostic contents\n")
        f.write("===========================\n\n")
        for name in ["backend", "iterations", "t", "avail_fields", "avail_species", "avail_record_components"]:
            try:
                value = getattr(ts, name)
            except Exception as exc:
                value = f"<unavailable: {exc}>"
            f.write(f"{name}:\n{repr(value)}\n\n")


def select_iterations(all_iterations: Iterable[int], requested: Optional[list[int]], stride: int) -> list[int]:
    all_iterations = list(map(int, all_iterations))
    if requested:
        available = set(all_iterations)
        selected = [int(it) for it in requested if int(it) in available]
        missing = sorted(set(map(int, requested)) - available)
        if missing:
            print(f"[warn] requested iterations not found and skipped: {missing}")
        return selected
    stride = max(1, int(stride))
    return all_iterations[::stride]


def time_for_iteration(ts: Any, iteration: int) -> float:
    try:
        iterations = np.asarray(ts.iterations, dtype=int)
        times = np.asarray(ts.t, dtype=float)
        idx = int(np.where(iterations == int(iteration))[0][0])
        return float(times[idx])
    except Exception:
        return float("nan")


def is_probably_available_field(ts: Any, field: str) -> bool:
    try:
        avail = getattr(ts, "avail_fields")
        if avail is None:
            return True
        return field in list(avail)
    except Exception:
        return True


def safe_get_field(ts: Any, iteration: int, requests: list[FieldRequest]) -> tuple[Optional[np.ndarray], Optional[Any], Optional[str]]:
    last_error = None
    for req in requests:
        if not is_probably_available_field(ts, req.field):
            continue
        try:
            kwargs = dict(req.kwargs)
            if req.coord is None:
                arr, info = ts.get_field(iteration=iteration, field=req.field, **kwargs)
            else:
                arr, info = ts.get_field(iteration=iteration, field=req.field, coord=req.coord, **kwargs)
            arr = np.asarray(arr)
            if np.iscomplexobj(arr):
                arr = np.real(arr)
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                continue
            return arr, info, req.label
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        pass
    return None, None, None


def axis_array(info: Any, names: list[str], n: Optional[int] = None) -> tuple[Optional[np.ndarray], Optional[str]]:
    for name in names:
        try:
            if hasattr(info, name):
                values = np.asarray(getattr(info, name), dtype=float)
                if values.ndim == 1 and (n is None or len(values) == n):
                    return values, name
        except Exception:
            continue
    return None, None


def prepare_2d_for_z_horizontal(data: np.ndarray, info: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Return array, z-axis, transverse-axis with z horizontal when possible."""
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D field, got shape {arr.shape}")

    ny, nx = arr.shape
    z_x, _ = axis_array(info, ["z"], nx)
    z_y, _ = axis_array(info, ["z"], ny)

    trans_names = ["r", "x", "y"]
    trans_y, trans_name_y = axis_array(info, trans_names, ny)
    trans_x, trans_name_x = axis_array(info, trans_names, nx)

    if z_x is not None:
        x_axis = z_x
        if trans_y is not None:
            y_axis = trans_y
            y_label = trans_name_y or "transverse"
        else:
            y_axis = np.arange(ny)
            y_label = "index"
        return arr, x_axis, y_axis, "z", y_label

    if z_y is not None:
        arr = arr.T
        x_axis = z_y
        if trans_x is not None:
            y_axis = trans_x
            y_label = trans_name_x or "transverse"
        else:
            y_axis = np.arange(nx)
            y_label = "index"
        return arr, x_axis, y_axis, "z", y_label

    # Fallback: no reliable metadata, keep original orientation.
    return arr, np.arange(nx), np.arange(ny), "x-index", "y-index"


def robust_signed_norm(values: np.ndarray, *, linthresh_fraction: float = 1e-3) -> tuple[Any, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, -1.0, 1.0
    vmax = float(np.nanpercentile(np.abs(finite), 99.5))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax
    linthresh = vmax * linthresh_fraction
    return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax), vmin, vmax


def robust_linear_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [1.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    return float(vmin), float(vmax)


def plot_field_image(
    ax: plt.Axes,
    data: np.ndarray,
    info: Any,
    title: str,
    value_label: str,
    scale: float = 1.0,
    signed: bool = True,
    cmap: str = "RdBu_r",
) -> None:
    arr, z_axis, y_axis, _, y_name = prepare_2d_for_z_horizontal(data, info)
    arr = arr * scale
    z_mm = z_axis * 1e3 if z_axis.dtype.kind in "fc" else z_axis
    if y_name in ["r", "x", "y"]:
        y_scaled = y_axis * 1e6
        y_label = f"{y_name} [µm]"
    else:
        y_scaled = y_axis
        y_label = y_name
    extent = [float(np.nanmin(z_mm)), float(np.nanmax(z_mm)), float(np.nanmin(y_scaled)), float(np.nanmax(y_scaled))]
    if signed:
        norm, _, _ = robust_signed_norm(arr)
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=norm)
    else:
        vmin, vmax = robust_linear_limits(arr)
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(y_label)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(value_label)


def get_on_axis_lineout(data: np.ndarray, info: Any, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, str]:
    arr, z_axis, y_axis, _, y_name = prepare_2d_for_z_horizontal(data, info)
    if y_axis.size:
        idx = int(np.nanargmin(np.abs(y_axis)))
    else:
        idx = 0
    line = arr[idx, :] * scale
    return z_axis * 1e3, line, y_name


def save_lineout(
    data: Optional[np.ndarray],
    info: Optional[Any],
    iteration: int,
    out_path: Path,
    title: str,
    ylabel: str,
    scale: float,
) -> None:
    if data is None or info is None:
        return
    try:
        z_mm, line, y_name = get_on_axis_lineout(data, info, scale=scale)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(z_mm, line)
        ax.set_title(f"{title}, iteration {iteration}")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] failed lineout {title} iteration {iteration}: {exc}")


def sample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def get_particles(ts: Any, iteration: int, species: str) -> Optional[dict[str, np.ndarray]]:
    try:
        avail_species = getattr(ts, "avail_species", None)
        if avail_species is not None and species not in list(avail_species):
            print(f"[warn] species {species!r} not found. Available species: {avail_species}")
            return None
    except Exception:
        pass

    # Preferred full particle record for this FBPIC ParticleDiagnostic.
    candidates = [
        ["z", "x", "y", "ux", "uy", "uz", "w"],
        ["z", "x", "y", "ux", "uy", "uz"],
        ["z", "ux", "uy", "uz", "w"],
        ["z", "uz", "w"],
        ["z", "uz"],
    ]
    last_error = None
    for var_list in candidates:
        try:
            arrays = ts.get_particle(var_list=var_list, iteration=iteration, species=species)
            if not isinstance(arrays, (list, tuple)):
                arrays = [arrays]
            out = {name: np.asarray(values) for name, values in zip(var_list, arrays)}
            n = len(next(iter(out.values())))
            if "w" not in out:
                out["w"] = np.ones(n)
            for name in ["x", "y", "ux", "uy"]:
                if name not in out:
                    out[name] = np.zeros(n)
            return out
        except Exception as exc:
            last_error = exc
    print(f"[warn] could not read particles for iteration {iteration}: {last_error}")
    return None


def particle_energy_MeV(p: dict[str, np.ndarray]) -> np.ndarray:
    ux = np.asarray(p.get("ux", 0.0), dtype=float)
    uy = np.asarray(p.get("uy", 0.0), dtype=float)
    uz = np.asarray(p.get("uz", 0.0), dtype=float)
    gamma = np.sqrt(1.0 + ux * ux + uy * uy + uz * uz)
    return (gamma - 1.0) * MEC2_MEV


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    mu = weighted_mean(values, weights)
    if not np.isfinite(mu):
        return float("nan")
    return math.sqrt(max(0.0, weighted_mean((values - mu) ** 2, weights)))


def weighted_cov(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    ma = weighted_mean(a, w)
    mb = weighted_mean(b, w)
    if not np.isfinite(ma) or not np.isfinite(mb):
        return float("nan")
    return weighted_mean((a - ma) * (b - mb), w)


def normalized_emittance(x: np.ndarray, ux: np.ndarray, w: np.ndarray) -> float:
    var_x = weighted_cov(x, x, w)
    var_ux = weighted_cov(ux, ux, w)
    cov = weighted_cov(x, ux, w)
    if not all(np.isfinite(v) for v in [var_x, var_ux, cov]):
        return float("nan")
    return math.sqrt(max(0.0, var_x * var_ux - cov * cov))


def plot_particle_panels(
    p: Optional[dict[str, np.ndarray]],
    iteration: int,
    out_particles: Path,
    plot_max_particles: int,
    bins: int,
    rng: np.random.Generator,
) -> None:
    if p is None:
        return
    n_total = len(p["z"])
    if n_total == 0:
        return
    idx = sample_indices(n_total, plot_max_particles, rng)
    z_mm = p["z"][idx] * 1e3
    energy = particle_energy_MeV(p)[idx]
    w = p["w"][idx]
    x_um = p["x"][idx] * 1e6
    y_um = p["y"][idx] * 1e6
    uz = p["uz"][idx]

    # 1) z-energy phase space
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    finite = np.isfinite(z_mm) & np.isfinite(energy) & np.isfinite(w) & (w > 0)
    if np.any(finite):
        hist = ax.hist2d(z_mm[finite], energy[finite], bins=bins, weights=w[finite], norm=LogNorm())
        cbar = plt.colorbar(hist[3], ax=ax, pad=0.02)
        cbar.set_label("weighted macroparticle count")
    ax.set_title(f"Electron phase space: z-energy, iteration {iteration}")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("kinetic energy [MeV]")
    fig.tight_layout()
    fig.savefig(out_particles / f"phase_z_energy_{iteration:08d}.png")
    plt.close(fig)

    # 2) energy spectrum, weighted by macro-particle weight
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if np.any(finite):
        ax.hist(energy[finite], bins=bins, weights=w[finite], histtype="step")
    ax.set_yscale("log")
    ax.set_title(f"Energy spectrum, iteration {iteration}")
    ax.set_xlabel("kinetic energy [MeV]")
    ax.set_ylabel("weighted macroparticle count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_particles / f"energy_spectrum_{iteration:08d}.png")
    plt.close(fig)

    # 3) z-uz phase space
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    finite2 = np.isfinite(z_mm) & np.isfinite(uz) & np.isfinite(w) & (w > 0)
    if np.any(finite2):
        hist = ax.hist2d(z_mm[finite2], uz[finite2], bins=bins, weights=w[finite2], norm=LogNorm())
        cbar = plt.colorbar(hist[3], ax=ax, pad=0.02)
        cbar.set_label("weighted macroparticle count")
    ax.set_title(f"Electron longitudinal phase space: z-uz, iteration {iteration}")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("uz = pz/(mec)")
    fig.tight_layout()
    fig.savefig(out_particles / f"phase_z_uz_{iteration:08d}.png")
    plt.close(fig)

    # 4) transverse distribution
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    finite3 = np.isfinite(x_um) & np.isfinite(y_um) & np.isfinite(w) & (w > 0)
    if np.any(finite3):
        hist = ax.hist2d(x_um[finite3], y_um[finite3], bins=bins, weights=w[finite3], norm=LogNorm())
        cbar = plt.colorbar(hist[3], ax=ax, pad=0.02)
        cbar.set_label("weighted macroparticle count")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Transverse electron distribution, iteration {iteration}")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.tight_layout()
    fig.savefig(out_particles / f"transverse_xy_{iteration:08d}.png")
    plt.close(fig)


def plot_phase_space_on_axis(ax: plt.Axes, p: Optional[dict[str, np.ndarray]], plot_max_particles: int, bins: int, rng: np.random.Generator) -> None:
    if p is None or len(p.get("z", [])) == 0:
        ax.set_title("z-energy phase space: unavailable")
        ax.axis("off")
        return
    n_total = len(p["z"])
    idx = sample_indices(n_total, plot_max_particles, rng)
    z_mm = p["z"][idx] * 1e3
    energy = particle_energy_MeV(p)[idx]
    w = p["w"][idx]
    finite = np.isfinite(z_mm) & np.isfinite(energy) & np.isfinite(w) & (w > 0)
    if np.any(finite):
        hist = ax.hist2d(z_mm[finite], energy[finite], bins=bins, weights=w[finite], norm=LogNorm())
        plt.colorbar(hist[3], ax=ax, pad=0.02, label="weighted count")
    ax.set_title("electron z-energy phase space")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("kinetic energy [MeV]")


def field_request_sets() -> dict[str, list[FieldRequest]]:
    return {
        "electron_rho": [
            FieldRequest("rho_electrons", None, "rho_electrons", {"m": 0}),
            FieldRequest("rho_electrons", None, "rho_electrons", {}),
            FieldRequest("rho-electrons", None, "rho-electrons", {"m": 0}),
            FieldRequest("rho-electrons", None, "rho-electrons", {}),
            FieldRequest("rho", None, "rho", {"m": 0}),
            FieldRequest("rho", None, "rho", {}),
        ],
        "rho": [
            FieldRequest("rho", None, "rho", {"m": 0}),
            FieldRequest("rho", None, "rho", {}),
        ],
        "Ez": [
            FieldRequest("E", "z", "Ez, m=0", {"m": 0}),
            FieldRequest("E", "z", "Ez, theta=0", {"theta": 0.0}),
            FieldRequest("E", "z", "Ez", {}),
        ],
        "Etrans": [
            FieldRequest("E", "x", "Ex, theta=0", {"theta": 0.0}),
            FieldRequest("E", "r", "Er, m=0", {"m": 0}),
            FieldRequest("E", "r", "Er, theta=0", {"theta": 0.0}),
            FieldRequest("E", "y", "Ey, theta=0", {"theta": 0.0}),
            FieldRequest("E", "t", "Et, theta=0", {"theta": 0.0}),
        ],
    }


def plot_quicklook(
    ts: Any,
    iteration: int,
    out_frame: Path,
    p: Optional[dict[str, np.ndarray]],
    plot_max_particles: int,
    bins: int,
    rng: np.random.Generator,
    dpi: int,
) -> tuple[Optional[np.ndarray], Optional[Any], Optional[np.ndarray], Optional[Any], Optional[np.ndarray], Optional[Any]]:
    reqs = field_request_sets()
    rho_e, info_rho_e, label_rho_e = safe_get_field(ts, iteration, reqs["electron_rho"])
    rho, info_rho, label_rho = safe_get_field(ts, iteration, reqs["rho"])
    Ez, info_Ez, label_Ez = safe_get_field(ts, iteration, reqs["Ez"])
    Et, info_Et, label_Et = safe_get_field(ts, iteration, reqs["Etrans"])

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    t_fs = time_for_iteration(ts, iteration) * 1e15
    fig.suptitle(f"LWFA quicklook | iteration {iteration} | t = {t_fs:.2f} fs", fontsize=14)

    if rho_e is not None and info_rho_e is not None:
        # Convert electron charge density to approximate electron number density.
        plot_field_image(
            axes[0, 0],
            -rho_e / elementary_charge,
            info_rho_e,
            f"electron density from {label_rho_e}",
            "ne [m$^{-3}$]",
            scale=1.0 / 1e24,
            signed=False,
            cmap="viridis",
        )
    else:
        axes[0, 0].set_title("electron density: unavailable")
        axes[0, 0].axis("off")

    if Ez is not None and info_Ez is not None:
        plot_field_image(axes[0, 1], Ez, info_Ez, f"longitudinal field {label_Ez}", "Ez [GV/m]", scale=1e-9, signed=True)
    else:
        axes[0, 1].set_title("Ez: unavailable")
        axes[0, 1].axis("off")

    if Et is not None and info_Et is not None:
        plot_field_image(axes[1, 0], Et, info_Et, f"transverse laser/wake field {label_Et}", "E⊥ [TV/m]", scale=1e-12, signed=True)
    else:
        axes[1, 0].set_title("transverse E: unavailable")
        axes[1, 0].axis("off")

    plot_phase_space_on_axis(axes[1, 1], p, plot_max_particles, bins, rng)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_frame, dpi=dpi)
    plt.close(fig)
    return rho_e, info_rho_e, Ez, info_Ez, Et, info_Et


def collect_summary_row(
    ts: Any,
    iteration: int,
    species: str,
    bunch_threshold_MeV: float,
    charge_thresholds_MeV: list[float],
    lambda0_m: Optional[float],
) -> dict[str, float]:
    row: dict[str, float] = {}
    t_s = time_for_iteration(ts, iteration)
    row["iteration"] = float(iteration)
    row["time_fs"] = t_s * 1e15
    row["ct_mm"] = c * t_s * 1e3 if np.isfinite(t_s) else float("nan")

    # Field metrics
    reqs = field_request_sets()
    Ez, _, _ = safe_get_field(ts, iteration, reqs["Ez"])
    Et, _, _ = safe_get_field(ts, iteration, reqs["Etrans"])
    row["peak_abs_Ez_GVm"] = float(np.nanmax(np.abs(Ez)) * 1e-9) if Ez is not None else float("nan")
    row["peak_abs_Etrans_TVm"] = float(np.nanmax(np.abs(Et)) * 1e-12) if Et is not None else float("nan")
    if Et is not None and lambda0_m is not None and lambda0_m > 0:
        omega0 = 2.0 * np.pi * c / lambda0_m
        row["a0_est_from_Etrans"] = float(elementary_charge * np.nanmax(np.abs(Et)) / (m_e * c * omega0))
    else:
        row["a0_est_from_Etrans"] = float("nan")

    p = get_particles(ts, iteration, species)
    if p is None or len(p["z"]) == 0:
        particle_keys = [
            "n_macroparticles",
            "total_weighted_charge_pC",
            "max_energy_MeV",
            "mean_energy_gt_threshold_MeV",
            "rms_energy_gt_threshold_MeV",
            "rel_energy_spread_gt_threshold_percent",
            "sigma_z_gt_threshold_um",
            "sigma_r_gt_threshold_um",
            "emit_nx_gt_threshold_mm_mrad",
            "emit_ny_gt_threshold_mm_mrad",
        ]
        for key in particle_keys:
            row[key] = float("nan")
        for thr in charge_thresholds_MeV:
            row[f"charge_gt_{thr:g}_MeV_pC"] = float("nan")
        return row

    energy = particle_energy_MeV(p)
    w = np.asarray(p["w"], dtype=float)
    x = np.asarray(p["x"], dtype=float)
    y = np.asarray(p["y"], dtype=float)
    z = np.asarray(p["z"], dtype=float)
    ux = np.asarray(p["ux"], dtype=float)
    uy = np.asarray(p["uy"], dtype=float)
    mask_valid = np.isfinite(energy) & np.isfinite(w) & (w > 0)

    row["n_macroparticles"] = float(len(energy))
    row["total_weighted_charge_pC"] = float(elementary_charge * np.nansum(w[mask_valid]) * 1e12)
    row["max_energy_MeV"] = float(np.nanmax(energy[mask_valid])) if np.any(mask_valid) else float("nan")

    for thr in charge_thresholds_MeV:
        mask_thr = mask_valid & (energy >= float(thr))
        row[f"charge_gt_{thr:g}_MeV_pC"] = float(elementary_charge * np.nansum(w[mask_thr]) * 1e12)

    mask_bunch = mask_valid & (energy >= bunch_threshold_MeV)
    if np.any(mask_bunch):
        wb = w[mask_bunch]
        Eb = energy[mask_bunch]
        row["mean_energy_gt_threshold_MeV"] = weighted_mean(Eb, wb)
        row["rms_energy_gt_threshold_MeV"] = weighted_rms(Eb, wb)
        if row["mean_energy_gt_threshold_MeV"] > 0:
            row["rel_energy_spread_gt_threshold_percent"] = 100.0 * row["rms_energy_gt_threshold_MeV"] / row["mean_energy_gt_threshold_MeV"]
        else:
            row["rel_energy_spread_gt_threshold_percent"] = float("nan")
        row["sigma_z_gt_threshold_um"] = weighted_rms(z[mask_bunch], wb) * 1e6
        r = np.sqrt(x[mask_bunch] ** 2 + y[mask_bunch] ** 2)
        row["sigma_r_gt_threshold_um"] = weighted_rms(r, wb) * 1e6
        row["emit_nx_gt_threshold_mm_mrad"] = normalized_emittance(x[mask_bunch], ux[mask_bunch], wb) * 1e6
        row["emit_ny_gt_threshold_mm_mrad"] = normalized_emittance(y[mask_bunch], uy[mask_bunch], wb) * 1e6
    else:
        row["mean_energy_gt_threshold_MeV"] = float("nan")
        row["rms_energy_gt_threshold_MeV"] = float("nan")
        row["rel_energy_spread_gt_threshold_percent"] = float("nan")
        row["sigma_z_gt_threshold_um"] = float("nan")
        row["sigma_r_gt_threshold_um"] = float("nan")
        row["emit_nx_gt_threshold_mm_mrad"] = float("nan")
        row["emit_ny_gt_threshold_mm_mrad"] = float("nan")
    return row


def write_summary_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    # Preserve all later keys, if any.
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rows_to_arrays(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    return {key: np.asarray([row.get(key, float("nan")) for row in rows], dtype=float) for key in keys}


def save_series_plot(
    x: np.ndarray,
    series: list[tuple[np.ndarray, str]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for y, label in series:
        if np.all(~np.isfinite(y)):
            continue
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    ax.set_yscale(yscale)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_summary(rows: list[dict[str, float]], out_dir: Path) -> None:
    if not rows:
        return
    arr = rows_to_arrays(rows)
    x = arr.get("ct_mm", arr.get("time_fs", np.arange(len(rows))))
    xlabel = "propagation distance ct [mm]" if "ct_mm" in arr else "time [fs]"

    save_series_plot(
        x,
        [
            (arr.get("max_energy_MeV", np.full_like(x, np.nan)), "max energy"),
            (arr.get("mean_energy_gt_threshold_MeV", np.full_like(x, np.nan)), "mean energy above threshold"),
        ],
        xlabel,
        "energy [MeV]",
        "Accelerated-electron energy evolution",
        out_dir / "energy_evolution.png",
    )

    charge_series = []
    for key in sorted([k for k in arr.keys() if k.startswith("charge_gt_") and k.endswith("_MeV_pC")]):
        label = key.replace("charge_gt_", ">").replace("_MeV_pC", " MeV")
        charge_series.append((arr[key], label))
    if charge_series:
        save_series_plot(
            x,
            charge_series,
            xlabel,
            "charge [pC]",
            "Accelerated charge versus distance",
            out_dir / "charge_evolution.png",
            yscale="linear",
        )

    save_series_plot(
        x,
        [
            (arr.get("peak_abs_Ez_GVm", np.full_like(x, np.nan)), "|Ez|max"),
            (arr.get("peak_abs_Etrans_TVm", np.full_like(x, np.nan)), "|E⊥|max [TV/m]"),
        ],
        xlabel,
        "field amplitude [mixed units]",
        "Peak field evolution",
        out_dir / "field_evolution.png",
    )

    if "a0_est_from_Etrans" in arr and not np.all(~np.isfinite(arr["a0_est_from_Etrans"])):
        save_series_plot(
            x,
            [(arr["a0_est_from_Etrans"], "estimated a0")],
            xlabel,
            "a0",
            "Estimated laser a0 from transverse field",
            out_dir / "a0_evolution.png",
        )

    save_series_plot(
        x,
        [
            (arr.get("sigma_z_gt_threshold_um", np.full_like(x, np.nan)), "σz"),
            (arr.get("sigma_r_gt_threshold_um", np.full_like(x, np.nan)), "σr"),
        ],
        xlabel,
        "rms size [µm]",
        "Accelerated-bunch size evolution",
        out_dir / "bunch_size_evolution.png",
    )

    save_series_plot(
        x,
        [(arr.get("rel_energy_spread_gt_threshold_percent", np.full_like(x, np.nan)), "relative rms spread")],
        xlabel,
        "energy spread [%]",
        "Accelerated-bunch relative energy spread",
        out_dir / "energy_spread_evolution.png",
    )

    save_series_plot(
        x,
        [
            (arr.get("emit_nx_gt_threshold_mm_mrad", np.full_like(x, np.nan)), "εnx"),
            (arr.get("emit_ny_gt_threshold_mm_mrad", np.full_like(x, np.nan)), "εny"),
        ],
        xlabel,
        "normalized emittance [mm mrad]",
        "Accelerated-bunch normalized emittance",
        out_dir / "emittance_evolution.png",
    )


def plot_nominal_profile(module: Optional[Any], out_dir: Path) -> Optional[float]:
    if module is None:
        return None
    lambda0 = getattr(module, "lambda0", None)
    try:
        dens_func = getattr(module, "dens_func")
        p_zmax = float(getattr(module, "p_zmax", 0.003))
        n_e = float(getattr(module, "n_e", 1.0))
        center_left = float(getattr(module, "center_left", np.nan))
        center_right = float(getattr(module, "center_right", np.nan))
        zfoc = float(getattr(module, "zfoc", np.nan))
        z0 = float(getattr(module, "z0", np.nan))
        z = np.linspace(-0.1e-3, p_zmax + 0.1e-3, 2000)
        r = np.zeros_like(z)
        rel = dens_func(z, r)
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.plot(z * 1e3, rel * n_e / 1e24)
        for pos, label in [(center_left, "up-ramp center"), (center_right, "down-ramp center"), (zfoc, "laser focus"), (z0, "initial laser centroid")]:
            if np.isfinite(pos):
                ax.axvline(pos * 1e3, linestyle="--", linewidth=1.0, alpha=0.65)
                ax.text(pos * 1e3, ax.get_ylim()[1] * 0.82, label, rotation=90, va="top", ha="right", fontsize=8)
        ax.set_title("Nominal plasma density profile from input script")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("ne [10$^{24}$ m$^{-3}$]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "nominal_density_profile.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] could not plot nominal profile: {exc}")
    try:
        return float(lambda0) if lambda0 is not None else None
    except Exception:
        return None


def make_animation(frames_dir: Path, out_path: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        print(f"[warn] imageio not available; skip animation. Install with `pip install imageio`. Error: {exc}")
        return
    frames = sorted(frames_dir.glob("quicklook_*.png"))
    if not frames:
        print("[warn] no quicklook frames found; skip animation")
        return
    duration = 1.0 / max(float(fps), 0.1)
    with imageio.get_writer(out_path, mode="I", duration=duration) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    print(f"[ok] wrote animation: {out_path}")


def main() -> None:
    args = parse_args()
    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)
    dirs = mkdirs(out_dir)
    rng = np.random.default_rng(args.random_seed)

    OpenPMDTimeSeries = import_openpmd_viewer()
    if not diag_dir.exists():
        raise SystemExit(f"Diagnostic directory does not exist: {diag_dir}")
    ts = OpenPMDTimeSeries(str(diag_dir))
    write_available_contents(ts, dirs["root"] / "available_contents.txt")

    module = load_input_script(args.input_script)
    lambda0_m = plot_nominal_profile(module, dirs["setup"])

    all_iterations = list(map(int, ts.iterations))
    if not all_iterations:
        raise SystemExit("No openPMD iterations found.")
    frame_iterations = select_iterations(all_iterations, args.iterations, args.frame_stride)
    summary_iterations = select_iterations(all_iterations, None, args.summary_stride)

    print(f"[info] found {len(all_iterations)} iterations: {all_iterations[0]} ... {all_iterations[-1]}")
    print(f"[info] plotting {len(frame_iterations)} quicklook frames to {dirs['frames']}")
    print(f"[info] writing outputs under {out_dir.resolve()}")

    for i, iteration in enumerate(frame_iterations, start=1):
        print(f"[plot] {i}/{len(frame_iterations)} iteration {iteration}")
        p = None if args.skip_particle_plots else get_particles(ts, iteration, args.species)
        try:
            rho_e, info_rho_e, Ez, info_Ez, Et, info_Et = plot_quicklook(
                ts,
                iteration,
                dirs["frames"] / f"quicklook_{iteration:08d}.png",
                p,
                args.plot_max_particles,
                args.phase_bins,
                rng,
                args.dpi,
            )
            save_lineout(
                rho_e,
                info_rho_e,
                iteration,
                dirs["lineouts"] / f"electron_density_lineout_{iteration:08d}.png",
                "on-axis electron density",
                "ne [10$^{24}$ m$^{-3}$]",
                scale=-1.0 / elementary_charge / 1e24,
            )
            save_lineout(
                Ez,
                info_Ez,
                iteration,
                dirs["lineouts"] / f"Ez_lineout_{iteration:08d}.png",
                "on-axis longitudinal electric field",
                "Ez [GV/m]",
                scale=1e-9,
            )
            if not args.skip_particle_plots:
                plot_particle_panels(p, iteration, dirs["particles"], args.plot_max_particles, args.phase_bins, rng)
        except Exception as exc:
            print(f"[warn] failed plotting iteration {iteration}: {exc}")
            traceback.print_exc()

    if not args.skip_summary:
        print(f"[info] computing summary metrics for {len(summary_iterations)} iterations")
        rows: list[dict[str, float]] = []
        for i, iteration in enumerate(summary_iterations, start=1):
            print(f"[summary] {i}/{len(summary_iterations)} iteration {iteration}")
            try:
                row = collect_summary_row(
                    ts,
                    iteration,
                    args.species,
                    args.bunch_threshold_MeV,
                    args.charge_thresholds_MeV,
                    lambda0_m=lambda0_m,
                )
                rows.append(row)
            except Exception as exc:
                print(f"[warn] failed summary iteration {iteration}: {exc}")
                traceback.print_exc()
        write_summary_csv(rows, dirs["root"] / "summary.csv")
        plot_summary(rows, dirs["summary"])
        print(f"[ok] wrote summary: {dirs['root'] / 'summary.csv'}")

    if args.animate:
        make_animation(dirs["frames"], dirs["root"] / "quicklook.gif", args.fps)

    print("[done] visualization complete")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
