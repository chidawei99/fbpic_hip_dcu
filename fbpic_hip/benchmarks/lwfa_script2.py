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
Nz = 9600  # Number of gridpoints along z
zmax = 0.0  # Right end of the simulation box (meters)
zmin = -0.0004  # Left end of the simulation box (meters)
Nr = 512  # Number of gridpoints along r
rmax = 5e-05  # Length of the box along r (meters)
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

    # Run the simulation
    sim.step(1000)
    print('')