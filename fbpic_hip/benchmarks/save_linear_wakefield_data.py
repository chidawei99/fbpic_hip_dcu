#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save paper-ready numerical data for FBPIC's linear wakefield verification case.

This script follows tests/test_linear_wakefield.py from FBPIC and saves the
numerical fields instead of plotting them directly. Run the same script in the
Original, new-CUDA-backend, and DCU-backend environments separately.

Examples
--------
# Original FBPIC
python save_linear_wakefield_data.py --backend original

# New CUDA backend
python save_linear_wakefield_data.py --backend cuda

# DCU backend
python save_linear_wakefield_data.py --backend dcu

# Only run Nm=2
python save_linear_wakefield_data.py --backend dcu --nm 2

MPI launch is also supported, e.g.
mpirun -n 2 python save_linear_wakefield_data.py --backend dcu --nm 1 2 3

Output
------
paper_data_linear_wakefield/
  original/
    Nm1.npz
    Nm2.npz
    Nm3.npz
    summary.csv
    metadata.json
  cuda/
    ...
  dcu/
    ...

Each Nm*.npz contains:
  z_m, r_m, z_um, r_um
  Ez_analytical, Er_analytical
  Ez_sim, Er_sim
  Ez_difference, Er_difference
  Ez_modes, Er_modes
  lineout_index, lineout_r_m, lineout_r_um
  Ez_analytical_lineout, Ez_sim_lineout
  Er_analytical_lineout, Er_sim_lineout
  simulation_time_s, Nm, backend

Array convention
----------------
All reconstructed 2-D field arrays have shape (Nz, Nr), i.e.
field[iz, ir].  For plotting with matplotlib.imshow, use field.T.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.integrate import quad

from fbpic.main import Simulation
from fbpic.lpa_utils.laser import (
    add_laser_pulse,
    GaussianLaser,
    LaguerreGaussLaser,
)


# ---------------------------------------------------------------------------
# Physical/numerical parameters: kept consistent with FBPIC
# tests/test_linear_wakefield.py
# ---------------------------------------------------------------------------

USE_CUDA = True

# Simulation box
Nz = 800
zmax = 40.e-6
Nr = 120
rmax = 60.e-6

# Time step and number of steps
dt = zmax / Nz / c
N_step = 1500

# Plasma
p_zmin = 39.e-6
p_zmax = 41.e-6
p_rmin = 0.0
p_rmax = 55.e-6
n_e = 8.e24
p_nz = 2
p_nr = 2

# Laser
a0 = 0.01
w0 = 20.e-6
ctau = 6.e-6
tau = ctau / c
z0 = 22.e-6

# Wavenumbers
kp = (1.0 / c) * np.sqrt(n_e * e**2 / (m_e * epsilon_0))
k0 = 2 * np.pi / 0.8e-6


# ---------------------------------------------------------------------------
# Analytical solution -- copied in form from the official regression test
# ---------------------------------------------------------------------------

def kernel_Ez(xi0, xi):
    return np.cos(kp * (xi - xi0)) * np.exp(-2 * (xi0 - z0)**2 / ctau**2)


def kernel_Er(xi0, xi):
    return np.sin(kp * (xi - xi0)) * np.exp(-2 * (xi0 - z0)**2 / ctau**2)


def analytical_Ez(z, r, t, Nm):
    Nz_local = len(z)
    window_zmax = z.max()

    long_profile = np.zeros(Nz_local)
    for iz in range(Nz_local):
        xi = z[iz] - c * t
        long_profile[iz] = quad(
            kernel_Ez,
            xi,
            window_zmax - c * t,
            args=(xi,),
            limit=30,
        )[0]

    if Nm in (1, 3):
        trans_profile = 4 * (r / w0)**2 * np.exp(-2 * r**2 / w0**2)
    elif Nm == 2:
        trans_profile = np.exp(-2 * r**2 / w0**2)
    else:
        raise ValueError("Nm must be 1, 2, or 3.")

    return (
        m_e * c**2 * kp**2 * a0**2 / (4.0 * e)
        * trans_profile[np.newaxis, :]
        * long_profile[:, np.newaxis]
    )


def analytical_Er(z, r, t, Nm):
    Nz_local = len(z)
    window_zmax = z.max()

    long_profile = np.zeros(Nz_local)
    for iz in range(Nz_local):
        xi = z[iz] - c * t
        long_profile[iz] = quad(
            kernel_Er,
            xi,
            window_zmax - c * t,
            args=(xi,),
            limit=200,
        )[0]

    if Nm in (1, 3):
        trans_profile = (
            8 * (r / w0**2)
            * (1 - 2 * r**2 / w0**2)
            * np.exp(-2 * r**2 / w0**2)
        )
    elif Nm == 2:
        trans_profile = -4 * r / w0**2 * np.exp(-2 * r**2 / w0**2)
    else:
        raise ValueError("Nm must be 1, 2, or 3.")

    return (
        m_e * c**2 * kp * a0**2 / (4.0 * e)
        * trans_profile[np.newaxis, :]
        * long_profile[:, np.newaxis]
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def field_metrics(numerical, analytical, official_atol_fraction):
    diff = numerical - analytical

    ana_l2 = np.linalg.norm(analytical.ravel())
    ana_max = np.max(np.abs(analytical))

    l2_relative = (
        np.linalg.norm(diff.ravel()) / ana_l2
        if ana_l2 != 0 else np.nan
    )
    linf_normalized = (
        np.max(np.abs(diff)) / ana_max
        if ana_max != 0 else np.nan
    )
    rmse_normalized = (
        np.sqrt(np.mean(diff**2)) / ana_max
        if ana_max != 0 else np.nan
    )

    official_atol = official_atol_fraction * ana_max
    official_pass = np.allclose(
        numerical,
        analytical,
        atol=official_atol,
    )

    return {
        "relative_L2": float(l2_relative),
        "normalized_Linf": float(linf_normalized),
        "normalized_RMSE": float(rmse_normalized),
        "analytical_max_abs_V_per_m": float(ana_max),
        "difference_max_abs_V_per_m": float(np.max(np.abs(diff))),
        "official_atol_V_per_m": float(official_atol),
        "official_pass": bool(official_pass),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def make_laser_profile(Nm):
    if Nm == 1:
        # Azimuthally polarized annular pulse, identical to official test
        return (
            LaguerreGaussLaser(
                0, 1,
                a0=a0,
                waist=w0,
                tau=tau,
                z0=z0,
                theta_pol=np.pi / 2,
                theta0=0.0,
            )
            +
            LaguerreGaussLaser(
                0, 1,
                a0=a0,
                waist=w0,
                tau=tau,
                z0=z0,
                theta_pol=0.0,
                theta0=-np.pi / 2,
            )
        )

    if Nm == 2:
        return GaussianLaser(
            a0=a0,
            waist=w0,
            tau=tau,
            z0=z0,
            theta_pol=np.pi / 2,
        )

    if Nm == 3:
        return LaguerreGaussLaser(
            0, 1,
            a0=a0,
            waist=w0,
            tau=tau,
            z0=z0,
            theta_pol=np.pi / 2,
        )

    raise ValueError("Nm must be 1, 2, or 3.")


def run_case(Nm, backend_label, output_dir, lineout_index):
    p_nt = 2 * Nm

    sim = Simulation(
        Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax,
        p_rmin, p_rmax,
        p_nz, p_nr, p_nt,
        n_e,
        use_cuda=USE_CUDA,
        boundaries={"z": "open", "r": "reflective"},
    )

    profile = make_laser_profile(Nm)
    add_laser_pulse(sim, profile)

    sim.set_moving_window(v=c)

    # Keep official test behavior for MPI runs
    correct_currents = False if sim.comm.size > 1 else True
    sim.step(N_step, correct_currents=correct_currents)

    # Gather all azimuthal modes to rank 0
    gathered_grids = [
        sim.comm.gather_grid(sim.fld.interp[m])
        for m in range(Nm)
    ]

    if sim.comm.rank != 0:
        return None

    grid0 = gathered_grids[0]
    z = np.asarray(grid0.z)
    r = np.asarray(grid0.r)

    # Save full complex modal fields as well, so later figures are not limited
    # to the theta=0 reconstruction.
    Ez_modes = np.stack(
        [np.asarray(g.Ez).copy() for g in gathered_grids],
        axis=0,
    )
    Er_modes = np.stack(
        [np.asarray(g.Er).copy() for g in gathered_grids],
        axis=0,
    )

    # Reconstruct theta=0 plane exactly as in official test
    Ez_sim = Ez_modes[0].real.copy()
    Er_sim = Er_modes[0].real.copy()

    for m in range(1, Nm):
        Ez_sim += 2.0 * Ez_modes[m].real
        Er_sim += 2.0 * Er_modes[m].real

    Ez_ana = analytical_Ez(z, r, sim.time, Nm)
    Er_ana = analytical_Er(z, r, sim.time, Nm)

    Ez_diff = Ez_sim - Ez_ana
    Er_diff = Er_sim - Er_ana

    ez_metrics = field_metrics(Ez_sim, Ez_ana, official_atol_fraction=0.08)
    er_metrics = field_metrics(Er_sim, Er_ana, official_atol_fraction=0.11)

    # Keep the official plot's off-axis idea. Clip only to make CLI robust.
    ir = int(np.clip(lineout_index, 0, len(r) - 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"Nm{Nm}.npz"

    np.savez_compressed(
        npz_path,

        # Coordinates
        z_m=z,
        r_m=r,
        z_um=z * 1.e6,
        r_um=r * 1.e6,

        # Full reconstructed fields, shape = (Nz, Nr)
        Ez_analytical=Ez_ana,
        Er_analytical=Er_ana,
        Ez_sim=Ez_sim,
        Er_sim=Er_sim,
        Ez_difference=Ez_diff,
        Er_difference=Er_diff,

        # Full complex azimuthal modes, shape = (Nm, Nz, Nr)
        Ez_modes=Ez_modes,
        Er_modes=Er_modes,

        # Convenient lineout data
        lineout_index=np.int64(ir),
        lineout_r_m=np.float64(r[ir]),
        lineout_r_um=np.float64(r[ir] * 1.e6),
        Ez_analytical_lineout=Ez_ana[:, ir],
        Ez_sim_lineout=Ez_sim[:, ir],
        Er_analytical_lineout=Er_ana[:, ir],
        Er_sim_lineout=Er_sim[:, ir],

        # Run identifiers
        simulation_time_s=np.float64(sim.time),
        Nm=np.int64(Nm),
        backend=np.array(backend_label),
    )

    row = {
        "backend": backend_label,
        "Nm": Nm,
        "simulation_time_s": float(sim.time),
        "lineout_index": ir,
        "lineout_r_um": float(r[ir] * 1.e6),

        "Ez_relative_L2": ez_metrics["relative_L2"],
        "Ez_normalized_Linf": ez_metrics["normalized_Linf"],
        "Ez_normalized_RMSE": ez_metrics["normalized_RMSE"],
        "Ez_analytical_max_abs_V_per_m":
            ez_metrics["analytical_max_abs_V_per_m"],
        "Ez_difference_max_abs_V_per_m":
            ez_metrics["difference_max_abs_V_per_m"],
        "Ez_official_atol_V_per_m":
            ez_metrics["official_atol_V_per_m"],
        "Ez_official_pass": ez_metrics["official_pass"],

        "Er_relative_L2": er_metrics["relative_L2"],
        "Er_normalized_Linf": er_metrics["normalized_Linf"],
        "Er_normalized_RMSE": er_metrics["normalized_RMSE"],
        "Er_analytical_max_abs_V_per_m":
            er_metrics["analytical_max_abs_V_per_m"],
        "Er_difference_max_abs_V_per_m":
            er_metrics["difference_max_abs_V_per_m"],
        "Er_official_atol_V_per_m":
            er_metrics["official_atol_V_per_m"],
        "Er_official_pass": er_metrics["official_pass"],
    }

    print(f"[saved] {npz_path}")
    print(
        f"        Ez: L2={row['Ez_relative_L2']:.6e}, "
        f"Linf={row['Ez_normalized_Linf']:.6e}, "
        f"official_pass={row['Ez_official_pass']}"
    )
    print(
        f"        Er: L2={row['Er_relative_L2']:.6e}, "
        f"Linf={row['Er_normalized_Linf']:.6e}, "
        f"official_pass={row['Er_official_pass']}"
    )

    return row


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------

def write_summary_csv(rows, path):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path, backend_label, nm_values, lineout_index):
    metadata = {
        "case": "FBPIC tests/test_linear_wakefield.py",
        "purpose": "Numerical verification data for paper figures",
        "backend_label": backend_label,
        "Nm_values": list(nm_values),
        "array_convention": "2-D field arrays use shape (Nz, Nr)",
        "theta_plane_for_reconstructed_fields_rad": 0.0,
        "lineout_index_requested": int(lineout_index),

        "simulation": {
            "Nz": Nz,
            "Nr": Nr,
            "zmax_m": zmax,
            "rmax_m": rmax,
            "dt_s": dt,
            "N_step": N_step,
            "use_cuda": USE_CUDA,
            "boundaries": {"z": "open", "r": "reflective"},
        },

        "plasma": {
            "p_zmin_m": p_zmin,
            "p_zmax_m": p_zmax,
            "p_rmin_m": p_rmin,
            "p_rmax_m": p_rmax,
            "density_m^-3": n_e,
            "p_nz": p_nz,
            "p_nr": p_nr,
            "p_nt_rule": "2 * Nm",
        },

        "laser": {
            "a0": a0,
            "waist_m": w0,
            "ctau_m": ctau,
            "tau_s": tau,
            "z0_m": z0,
            "lambda0_m": 0.8e-6,
            "k0_m^-1": k0,
        },

        "plasma_wavenumber_kp_m^-1": kp,

        "official_test_tolerances": {
            "Ez_atol": "0.08 * max(abs(Ez_analytical))",
            "Er_atol": "0.11 * max(abs(Er_analytical))",
        },

        "recommended_paper_figures": [
            "Nm=2: Analytical / Original / CUDA / DCU Ez cloud maps",
            "Nm=2: Analytical / Original / CUDA / DCU Er cloud maps",
            "Nm=2: Ez and Er off-axis lineouts",
            "Nm=2: CUDA-Original and DCU-Original normalized difference maps",
            "Nm=1,2,3: quantitative error table from summary.csv",
        ],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save FBPIC linear-wakefield data for paper figures."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["original", "cuda", "dcu"],
        help=(
            "Label only. Run this script inside the corresponding FBPIC "
            "environment/repository."
        ),
    )
    parser.add_argument(
        "--nm",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Azimuthal mode counts to run. Default: 1 2 3",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_data_linear_wakefield",
        help="Root output directory.",
    )
    parser.add_argument(
        "--lineout-index",
        type=int,
        default=10,
        help=(
            "Radial grid index used for convenient off-axis lineout data. "
            "Default 10 matches the official test plotting choice."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    backend_dir = Path(args.output_dir) / args.backend

    rows = []
    for Nm in args.nm:
        row = run_case(
            Nm=Nm,
            backend_label=args.backend,
            output_dir=backend_dir,
            lineout_index=args.lineout_index,
        )
        if row is not None:
            rows.append(row)

    # Only rank 0 receives rows
    if rows:
        write_summary_csv(rows, backend_dir / "summary.csv")
        write_metadata(
            backend_dir / "metadata.json",
            backend_label=args.backend,
            nm_values=args.nm,
            lineout_index=args.lineout_index,
        )

        print(f"[saved] {backend_dir / 'summary.csv'}")
        print(f"[saved] {backend_dir / 'metadata.json'}")
        print("[done] Paper-ready numerical data saved successfully.")


if __name__ == "__main__":
    main()
