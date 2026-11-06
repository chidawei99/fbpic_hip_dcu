"""
This is an input script that runs a simulation of
laser-wakefield acceleration using FBPIC.
Usage
-----
- Modify the parameters below to suit your needs
- Type "python lwfa_script.py" in a terminal
Help
----
All the structures implemented in FBPIC are internally documented.
Enter "print(fbpic_object.__doc__)" to have access to this documentation,
where fbpic_object is any of the objects or function of FBPIC.
"""

# -------
# Imports
# -------

import argparse
import csv
import os
from pathlib import Path

import numpy as np
# Import the relevant structures in FBPIC
from fbpic.lpa_utils.laser import add_laser_pulse, FlattenedGaussianLaser
from fbpic.main import Simulation
from fbpic.openpmd_diag import (
    FieldDiagnostic,
    ParticleDiagnostic,
    ParticleChargeDensityDiagnostic,
)
from scipy.constants import c, e, m_e

# ----------
# Parameters
# ----------

# Whether to use the GPU
use_cuda = True

# Order of the stencil for z derivatives in the Maxwell solver.
# Use -1 for infinite order, i.e. for exact dispersion relation in
# all direction (adviced for single-GPU/single-CPU simulation).
# Use a positive number (and multiple of 2) for a finite-order stencil
# (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
# to more overhead in MPI communications, but also to a more accurate
# dispersion relation for electromagnetic waves. (Typically,
# `n_order = 32` is a good trade-off.)
# See https://arxiv.org/abs/1611.05712 for more information.
n_order = 32

# The simulation box
Nz = 2453  # Number of gridpoints along z
zmax = 0.0  # Right end of the simulation box (meters)
zmin = -0.0001  # Left end of the simulation box (meters)
Nr = 1024  # Number of gridpoints along r
rmax = 5e-04  # Length of the box along r (meters)
Nm = 3  # Number of modes used

# The simulation timestep
dt = (zmax - zmin) / Nz / c  # Timestep (seconds)

# The particles
p_zmin = 0.0  # Position of the beginning of the plasma (meters)
p_zmax = 0.003  # Position of the end of the plasma (meters)
p_rmax = 5e-05  # Maximal radial position of the plasma (meters)
n_e = 8e+24  # Density (electrons.meters^-3)
p_nz = 2  # Number of particles per cell along z
p_nr = 2  # Number of particles per cell along r
p_nt = 12  # Number of particles per cell along theta

# The laser
a0 = 2.4  # Laser amplitude
w0 = 1.868507960633642e-05  # Laser waist
z0 = -1e-05  # Laser centroid
lambda0 = 8.15e-07  # Laser wavelength
zfoc = 0.00010000000000000005  # Focal position
tau = 2.123304500720048e-14  # Laser duration
profile_flatness = 6  # Flatness of laser profile far from focus (larger means flatter)

# The moving window
v_window = c  # Speed of the window

# The diagnostics and the checkpoints/restarts
diag_period = 1000  # Period of the diagnostics in number of timesteps

# The density profile
flat_top_dist = 0.001  # plasma flat top distance
sigma_right = 0.0005
center_left = 0.001
sigma_left = 0.0005
power = 2.0
center_right = 0.002




def parse_benchmark_args(default_Nz):
    """Parse only the benchmark options that are allowed to vary."""
    parser = argparse.ArgumentParser(
        description="Run the LWFA FBPIC benchmark with fixed physics parameters."
    )
    parser.add_argument(
        "--Nz",
        "--nz",
        dest="Nz",
        type=int,
        default=None,
        help="Number of gridpoints along z. If omitted, read NZ from the environment; otherwise use the script default.",
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=("strong", "weak"),
        default="weak",
        help="Scaling benchmark type. Use 'strong' or 'weak'.",
    )
    args = parser.parse_args()

    env_Nz = os.environ.get("NZ")
    if args.Nz is not None:
        Nz_value = args.Nz
    elif env_Nz is not None:
        Nz_value = int(env_Nz)
    else:
        Nz_value = default_Nz

    if Nz_value <= 0:
        raise ValueError("Nz must be a positive integer.")

    args.Nz = Nz_value
    return args


def get_mpi_comm_rank_size():
    """Return MPI communicator, rank, and size; fall back to serial mode."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


def mpi_barrier(comm):
    if comm is not None:
        comm.Barrier()


def mpi_max(comm, value):
    if comm is None:
        return value
    from mpi4py import MPI
    return comm.allreduce(value, op=MPI.MAX)


def run_cupy_benchmark(sim, warmup_steps=20, timed_steps=500):
    """Benchmark sim.step with CuPy CUDA/HIP events and return seconds per step."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is required for this benchmark timing path.") from exc

    if warmup_steps > 0:
        sim.step(warmup_steps)

    cp.cuda.Device().synchronize()
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()

    start_event.record()
    sim.step(timed_steps)
    stop_event.record()
    stop_event.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start_event, stop_event)
    return elapsed_ms * 1.0e-3 / timed_steps


def save_benchmark_result(mpi_size, benchmark_kind, Nz, avg_step_time_s):
    """Save one benchmark result file in the current working directory."""
    output_path = Path.cwd() / f"{mpi_size}_{benchmark_kind}.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mpi_processes",
            "avg_step_time_s",
            "Nz",
            "benchmark_kind",
            "warmup_steps",
            "timed_steps",
        ])
        writer.writerow([
            mpi_size,
            f"{avg_step_time_s:.12e}",
            Nz,
            benchmark_kind,
            20,
            500,
        ])
    return output_path


def dens_func(z, r):
    """
    User-defined function: density profile of the plasma
    It should return the relative density with respect to n_plasma,
    at the position x, y, z (i.e. return a number between 0 and 1)
    Parameters
    ----------
    z, r: 1darrays of floats
        Arrays with one element per macroparticle
    Returns
    -------
    n : 1d array of floats
        Array of relative density, with one element per macroparticles
    """

    # Allocate relative density
    n = np.ones_like(z)

    # before up-ramp
    n = np.where(z < 0.0, 0.0, n)

    # Make up-ramp
    n = np.where(
        z < center_left,
        np.exp(-(((z - center_left) / sigma_left) ** power)),
        n,
    )

    # Make down-ramp
    n = np.where(
        (z >= center_right)
        & (z < center_right + 2 * sigma_right),
        np.exp(-(((z - center_right) / sigma_right) ** power)),
        n,
    )

    # after down-ramp
    n = np.where(z >= center_right + 2 * sigma_right, 0, n)

    return n


# The interaction length of the simulation (meters)
L_interact = 0.003  # increase to simulate longer distance!
# Interaction time (seconds) (to calculate number of PIC iterations)
T_interact = (L_interact + (zmax - zmin)) / v_window
# (i.e. the time it takes for the moving window to slide across the plasma)

# ---------------------------
# Carrying out the simulation
# ---------------------------

# NB: The code below is only executed when running the script,
# (`python lwfa_script.py`), but not when importing it (`import lwfa_script`).
if __name__ == '__main__':
    benchmark_args = parse_benchmark_args(default_Nz=Nz)
    Nz = benchmark_args.Nz
    benchmark_kind = benchmark_args.benchmark_kind
    dt = (zmax - zmin) / Nz / c

    mpi_comm, mpi_rank, mpi_size = get_mpi_comm_rank_size()

    # Initialize the simulation object
    sim = Simulation(
        Nz=Nz,
        zmax=zmax,
        Nr=Nr,
        rmax=rmax,
        Nm=Nm,
        dt=dt,
        zmin=zmin,
        boundaries={
            "z": "open",
            "r": "open", # "reflective",
        },
        n_order=n_order,
        use_cuda=use_cuda,
    )

    # Create the plasma electrons
    plasma_elec = sim.add_new_species(
        q=-e,
        m=m_e,
        n=n_e,
        dens_func=dens_func,
        p_zmin=p_zmin,
        p_zmax=p_zmax,
        p_rmax=p_rmax,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
    )

    # Load initial fields
    # Add a laser to the fields of the simulation
    profile = FlattenedGaussianLaser(
        a0=a0,
        w0=w0,
        tau=tau,
        z0=z0,
        N=profile_flatness,
        zf=zfoc,
        lambda0=lambda0,
    )
    add_laser_pulse(
        sim=sim,
        laser_profile=profile,
    )

    # Configure the moving window
    sim.set_moving_window(v=v_window)

    # Add diagnostics
    sim.diags = [
        FieldDiagnostic(
            period=diag_period,
            fldobject=sim.fld,
            comm=sim.comm,
            fieldtypes=["rho", "E"],
        ),
        ParticleDiagnostic(
            period=diag_period,
            species={"electrons": plasma_elec},
            comm=sim.comm,
        ),
        ParticleChargeDensityDiagnostic(
            period=diag_period,
            sim=sim,
            species={"electrons": plasma_elec},
        ),
    ]

    # Number of iterations to perform
    N_step = int(T_interact / sim.dt)

    # set deterministic random seed
    np.random.seed(42)

    # Run the benchmark
    local_avg_step_time_s = run_cupy_benchmark(
        sim,
        warmup_steps=20,
        timed_steps=500,
    )
    mpi_barrier(mpi_comm)
    avg_step_time_s = mpi_max(mpi_comm, local_avg_step_time_s)

    if mpi_rank == 0:
        output_path = save_benchmark_result(
            mpi_size=mpi_size,
            benchmark_kind=benchmark_kind,
            Nz=Nz,
            avg_step_time_s=avg_step_time_s,
        )
        print(
            f"mpi_processes={mpi_size}, Nz={Nz}, benchmark_kind={benchmark_kind}, "
            f"avg_step_time_s={avg_step_time_s:.12e}, output={output_path}"
        )
    print('')
