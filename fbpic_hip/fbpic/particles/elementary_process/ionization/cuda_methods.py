# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in particle ionization.

Apart from synthactic details, this file is very close to numba_methods.py
"""
from numba import cuda
from fbpic.utils.cuda import compile_cupy
from scipy.constants import c
# Import inline functions
from .inline_functions import get_ionization_probability, \
    get_E_amplitude, copy_ionized_electrons_batch
# Compile the inline functions for GPU
get_ionization_probability = cuda.jit( get_ionization_probability,
                                        device=True, inline=False )
get_E_amplitude = cuda.jit( get_E_amplitude,
                            device=True, inline=False )
copy_ionized_electrons_batch = cuda.jit( copy_ionized_electrons_batch,
                                            device=True, inline=False )

import cupy as cp

ionize_ions_cuda_c = cp.RawKernel(r'''
extern "C" {

__device__ __forceinline__
double get_ionization_probability(
    double E, double gamma,
    double prefactor, double power, double exp_prefactor)
{
    if (E == 0.0) return 0.0;

    double w_dtau =
        (1.0 / gamma) * prefactor *
        pow(E, power) * exp(exp_prefactor / E);

    return 1.0 - exp(-w_dtau);
}

__device__ __forceinline__
void get_E_amplitude(
    double ux, double uy, double uz,
    double Ex, double Ey, double Ez,
    double cBx, double cBy, double cBz,
    double* E_out, double* gamma_out)
{
    double u_dot_E = ux*Ex + uy*Ey + uz*Ez;
    double gamma = sqrt(1.0 + ux*ux + uy*uy + uz*uz);

    double E2 =
        - u_dot_E * u_dot_E
        + pow(gamma*Ex + uy*cBz - uz*cBy, 2)
        + pow(gamma*Ey + uz*cBx - ux*cBz, 2)
        + pow(gamma*Ez + ux*cBy - uy*cBx, 2);

    *E_out = sqrt(E2);
    *gamma_out = gamma;
}

__global__
void ionize_ions(
    int N_batch, int batch_size, int Ntot,
    int level_start, int level_max, int n_levels,
    long long* n_ionized, short* ionized_from, unsigned long long* ionization_level,
    float* random_draw,
    const double* adk_prefactor,
    const double* adk_power,
    const double* adk_exp_prefactor,
    const double* ux, const double* uy, const double* uz,
    const double* Ex, const double* Ey, const double* Ez,
    const double* Bx, const double* By, const double* Bz,
    const double* w, double* w_times_level, const double c)
{
    int i_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_batch >= N_batch) return;

    /* reset batch counters */
    for (int i_level = 0; i_level < n_levels; i_level++) {
        n_ionized[i_level * N_batch + i_batch] = 0;
    }

    int ip_start = i_batch * batch_size;
    int ip_end   = min((i_batch + 1) * batch_size, Ntot);

    for (int ip = ip_start; ip < ip_end; ip++) {

        int level = ionization_level[ip];
        if (level >= level_max) {
            ionized_from[ip] = -1;
            continue;
        }

        double E, gamma;
        get_E_amplitude(
            ux[ip], uy[ip], uz[ip],
            Ex[ip], Ey[ip], Ez[ip],
            c*Bx[ip], c*By[ip], c*Bz[ip],
            &E, &gamma
        );

        double p = get_ionization_probability(
            E, gamma,
            adk_prefactor[level],
            adk_power[level],
            adk_exp_prefactor[level]
        );

        if (random_draw[ip] < p) {
            int out_level = level - level_start;
            ionized_from[ip] = out_level;

            if (n_levels == 1) {
                n_ionized[i_batch] += 1;
            } else {
                n_ionized[out_level * N_batch + i_batch] += 1;
            }

            ionization_level[ip] += 1;
            w_times_level[ip] = w[ip] * ionization_level[ip];
        }
        else {
            ionized_from[ip] = -1;
        }
    }
}

} // extern "C"
''', 'ionize_ions')
copy_ionized_electrons_cuda_c = cp.RawKernel(r'''
extern "C" {

/* -------------------------------
   device function: batch version
   ------------------------------- */
__device__ void copy_ionized_electrons_batch(
    const int i_batch,
    const int batch_size,
    const int elec_old_Ntot,
    const int ion_Ntot,
    const long long* cumulative_n_ionized,
    const short* ionized_from,
    const int i_level,
    bool store_electrons_per_level,
    int N_batch,

    double* elec_x, double* elec_y, double* elec_z, double* elec_inv_gamma,
    double* elec_ux, double* elec_uy, double* elec_uz, double* elec_w,
    double* elec_Ex, double* elec_Ey, double* elec_Ez,
    double* elec_Bx, double* elec_By, double* elec_Bz,

    const double* ion_x, const double* ion_y, const double* ion_z,
    const double* ion_inv_gamma,
    const double* ion_ux, const double* ion_uy, const double* ion_uz,
    const double* ion_w,
    const double* ion_Ex, const double* ion_Ey, const double* ion_Ez,
    const double* ion_Bx, const double* ion_By, const double* ion_Bz
){
    long long elec_index =
        elec_old_Ntot + cumulative_n_ionized[i_level * N_batch + i_batch];

    int start = i_batch * batch_size;
    int end = start + batch_size;
    if (end > ion_Ntot) end = ion_Ntot;

    for (int ion_index = start; ion_index < end; ++ion_index) {

        int create_electron = 0;

        if ((!store_electrons_per_level) && ionized_from[ion_index] >= 0)
            create_electron = 1;

        if (store_electrons_per_level && ionized_from[ion_index] == i_level)
            create_electron = 1;

        if (create_electron) {
            elec_x[elec_index] = ion_x[ion_index];
            elec_y[elec_index] = ion_y[ion_index];
            elec_z[elec_index] = ion_z[ion_index];
            elec_ux[elec_index] = ion_ux[ion_index];
            elec_uy[elec_index] = ion_uy[ion_index];
            elec_uz[elec_index] = ion_uz[ion_index];
            elec_inv_gamma[elec_index] = ion_inv_gamma[ion_index];
            elec_w[elec_index] = ion_w[ion_index];
            elec_Ex[elec_index] = ion_Ex[ion_index];
            elec_Ey[elec_index] = ion_Ey[ion_index];
            elec_Ez[elec_index] = ion_Ez[ion_index];
            elec_Bx[elec_index] = ion_Bx[ion_index];
            elec_By[elec_index] = ion_By[ion_index];
            elec_Bz[elec_index] = ion_Bz[ion_index];
            elec_index+= 1;
        }
    }
}


/* -------------------------------
   global kernel: one thread = one batch
   ------------------------------- */
__global__ void copy_ionized_electrons_cuda(
    const int N_batch,
    const int batch_size,
    const int elec_old_Ntot,
    const int ion_Ntot,
    const long long* cumulative_n_ionized,
    const short* ionized_from,
    const int i_level,
    bool store_electrons_per_level,

    double* elec_x, double* elec_y, double* elec_z, double* elec_inv_gamma,
    double* elec_ux, double* elec_uy, double* elec_uz, double* elec_w,
    double* elec_Ex, double* elec_Ey, double* elec_Ez,
    double* elec_Bx, double* elec_By, double* elec_Bz,

    const double* ion_x, const double* ion_y, const double* ion_z,
    const double* ion_inv_gamma,
    const double* ion_ux, const double* ion_uy, const double* ion_uz,
    const double* ion_w,
    const double* ion_Ex, const double* ion_Ey, const double* ion_Ez,
    const double* ion_Bx, const double* ion_By, const double* ion_Bz,
    int NN
){
    int i_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_batch < N_batch) {
        copy_ionized_electrons_batch(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, ionized_from,
            i_level, store_electrons_per_level,NN,
            elec_x, elec_y, elec_z, elec_inv_gamma,
            elec_ux, elec_uy, elec_uz, elec_w,
            elec_Ex, elec_Ey, elec_Ez,
            elec_Bx, elec_By, elec_Bz,
            ion_x, ion_y, ion_z, ion_inv_gamma,
            ion_ux, ion_uy, ion_uz, ion_w,
            ion_Ex, ion_Ey, ion_Ez,
            ion_Bx, ion_By, ion_Bz
        );
    }
}

} // extern "C"
''', 'copy_ionized_electrons_cuda')

@compile_cupy
def ionize_ions_cuda( N_batch, batch_size, Ntot,
    level_start, level_max, n_levels,
    n_ionized, ionized_from, ionization_level, random_draw,
    adk_prefactor, adk_power, adk_exp_prefactor,
    ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, w, w_times_level ):
    """
    For each ion macroparticle, decide whether it is going to
    be further ionized during this timestep, based on the ADK rate.

    Increment the elements in `ionization_level` accordingly, and update
    `w_times_level` of the ions to take into account the change in level
    of the corresponding macroparticle.

    For the purpose of counting and creating the corresponding electrons,
    `ionized_from` (one element per macroparticle) is set to -1 at the position
    of the unionized ions, and to the level (before ionization) otherwise
    `n_ionized` (one element per batch, and per ionizable level that needs
    to be distinguished) counts the total number of ionized particles
    in the current batch.
    """
    # Loop over batches of particles
    i_batch = cuda.grid(1)
    if i_batch < N_batch:

        # Set the count of ionized particles in the batch to 0
        for i_level in range(n_levels):
            n_ionized[i_level, i_batch] = 0

        # Loop through the batch
        N_max = min( (i_batch+1)*batch_size, Ntot )
        for ip in range( i_batch*batch_size, N_max ):

            # Skip the ionization routine, if the maximal ionization level
            # has already been reached for this macroparticle
            level = ionization_level[ip]
            if level >= level_max:
                ionized_from[ip] = -1
                continue

            # Calculate the amplitude of the electric field,
            # in the frame of the electrons (device inline function)
            E, gamma = get_E_amplitude( ux[ip], uy[ip], uz[ip],
                    Ex[ip], Ey[ip], Ez[ip], c*Bx[ip], c*By[ip], c*Bz[ip] )
            # Get ADK rate (device inline function)
            p = get_ionization_probability( E, gamma,
              adk_prefactor[level], adk_power[level], adk_exp_prefactor[level])
            # Ionize particles
            if random_draw[ip] < p:
                # Set the corresponding flag and update particle count
                ionized_from[ip] = level-level_start
                if n_levels == 1:
                    # No need to distinguish ionization levels
                    n_ionized[0, i_batch] += 1
                else:
                    # Distinguish count for each ionizable level
                    n_ionized[level-level_start, i_batch] += 1
                # Update the ionization level and the corresponding weight
                ionization_level[ip] += 1
                w_times_level[ip] = w[ip] * ionization_level[ip]
            else:
                ionized_from[ip] = -1

@compile_cupy
def copy_ionized_electrons_cuda(
    N_batch, batch_size, elec_old_Ntot, ion_Ntot,
    cumulative_n_ionized, ionized_from,
    i_level, store_electrons_per_level,
    elec_x, elec_y, elec_z, elec_inv_gamma,
    elec_ux, elec_uy, elec_uz, elec_w,
    elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz,
    ion_x, ion_y, ion_z, ion_inv_gamma,
    ion_ux, ion_uy, ion_uz, ion_w,
    ion_Ex, ion_Ey, ion_Ez, ion_Bx, ion_By, ion_Bz ):
    """
    Create the new electrons by copying the properties (position, momentum,
    etc) of the ions that they originate from.
    """
    # Select the current batch
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        copy_ionized_electrons_batch(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, ionized_from,
            i_level, store_electrons_per_level,
            elec_x, elec_y, elec_z, elec_inv_gamma,
            elec_ux, elec_uy, elec_uz, elec_w,
            elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz,
            ion_x, ion_y, ion_z, ion_inv_gamma,
            ion_ux, ion_uy, ion_uz, ion_w,
            ion_Ex, ion_Ey, ion_Ez, ion_Bx, ion_By, ion_Bz )
