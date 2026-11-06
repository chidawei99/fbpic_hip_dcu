# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This files contains cuda methods that are used in the boosted-frame
diagnostics
"""
import numpy as np
from fbpic.utils.cuda import cupy, cuda_tpb_bpg_1d

def extract_slice_from_gpu( pref_sum_curr, N_area, species ):
    """
    Extract the particles which have which have index between pref_sum_curr
    and pref_sum_curr + N_area, and return them in dictionaries.

    Parameters
    ----------
    pref_sum_curr: int
        The starting index needed for the extraction process
    N_area: int
        The number of particles to extract.
    species: an fbpic Species object
        The species from to extract data

    Returns
    -------
    particle_data : A dictionary of 1D float arrays (that are on the CPU)
        A dictionary that contains the particle data of
        the simulation (with normalized weigths), including optional
        integer arrays (e.g. "id", "charge")
    """
    # Call kernel that extracts particles from GPU
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(N_area)
    # - General particle quantities
    part_data = cupy.empty( (8, N_area), dtype=np.float64 )
    extract_particles_from_gpu_c((dim_grid_1d,), (dim_block_1d,), ( pref_sum_curr,
         species.x, species.y, species.z, species.ux, species.uy, species.uz,
         species.w, species.inv_gamma, part_data, part_data.shape[1] ))
    # - Optional particle arrays
    if species.tracker is not None:
        selected_particle_id = cupy.empty( (N_area,), dtype=np.uint64 )
        extract_array_from_gpu_uint((dim_grid_1d,), (dim_block_1d,), (
            pref_sum_curr, species.tracker.id, selected_particle_id, selected_particle_id.shape[0] ))
    if species.ionizer is not None:
        selected_particle_charge = cupy.empty( (N_area,), dtype=np.uint64 )
        extract_array_from_gpu_uint((dim_grid_1d,), (dim_block_1d,), ( pref_sum_curr,
          species.ionizer.ionization_level, selected_particle_charge, selected_particle_charge.shape[0] ))
        selected_particle_weight = cupy.empty( (N_area,), dtype=np.float64 )
        extract_array_from_gpu_c((dim_grid_1d,), (dim_block_1d,), ( pref_sum_curr,
          species.ionizer.w_times_level, selected_particle_weight, selected_particle_weight.shape[0] ))

    # Copy GPU arrays to the host
    part_data = part_data.get()
    particle_data = { 'x':part_data[0], 'y':part_data[1], 'z':part_data[2],
        'ux':part_data[3], 'uy':part_data[4], 'uz':part_data[5],
        'w':part_data[6], 'inv_gamma':part_data[7] }
    if species.tracker is not None:
        particle_data['id'] = selected_particle_id.get()
    if species.ionizer is not None:
        particle_data['charge'] = selected_particle_charge.get()
        # Replace particle weight
        particle_data['w'] = selected_particle_weight.get()

    # Return the data as dictionary
    return( particle_data )


cuda_src = r'''
extern "C" __global__
void extract_particles_from_gpu(
    const int part_idx_start,
    const double*  x,
    const double*  y,
    const double*  z,
    const double*  ux,
    const double*  uy,
    const double*  uz,
    const double*  w,
    const double*  inv_gamma,
    double*  selected,
    const int N_part)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N_part) {
        int ptcl_idx = part_idx_start + i;

        // selected is stored in row-major (C-order): selected[8, N_part]
        // index = row * N_part + col
        selected[0 * N_part + i] = x[ptcl_idx];
        selected[1 * N_part + i] = y[ptcl_idx];
        selected[2 * N_part + i] = z[ptcl_idx];
        selected[3 * N_part + i] = ux[ptcl_idx];
        selected[4 * N_part + i] = uy[ptcl_idx];
        selected[5 * N_part + i] = uz[ptcl_idx];
        selected[6 * N_part + i] = w[ptcl_idx];
        selected[7 * N_part + i] = inv_gamma[ptcl_idx];
    }
}

extern "C" __global__
void extract_array_from_gpu(
    const int part_idx_start,
    const double*  array,
    double*  selected,
    const int N_part)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N_part) {
        int ptcl_idx = part_idx_start + i;
        selected[i] = array[ptcl_idx];
    }
}

extern "C" __global__
void extract_array_from_gpu_uint(
    const int part_idx_start,
    const unsigned long long*  array,
    unsigned long long*  selected,
    const int N_part)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N_part) {
        int ptcl_idx = part_idx_start + i;
        selected[i] = array[ptcl_idx];
    }
}
'''
extract_particles_from_gpu_c = cupy.RawKernel(cuda_src, 'extract_particles_from_gpu')
extract_array_from_gpu_c = cupy.RawKernel(cuda_src, 'extract_array_from_gpu')
extract_array_from_gpu_uint = cupy.RawKernel(cuda_src, 'extract_array_from_gpu_uint')



