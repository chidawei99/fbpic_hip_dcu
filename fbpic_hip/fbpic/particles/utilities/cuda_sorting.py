# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle sorting methods on the GPU using CUDA.
"""
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from cupy.cuda import thrust
    from fbpic.utils.printing import catch_gpu_memory_error
import math
import numpy as np

# -----------------------------------------------------
# Sorting utilities - get_cell_idx / sort / prefix_sum
# -----------------------------------------------------
import cupy as cp
src = r'''
extern "C" __global__
void get_cell_idx_per_particle(
    int* cell_idx,
    long long* sorted_idx,
    const double* x,
    const double* y,
    const double* z,
    const double invdz,
    const double zmin,
    const int Nz,
    const double invdr,
    const double rmin,
    const int Nr,
    int Ntot   // total number of particles
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Ntot) return;

    // Load particle positions
    double xj = x[i];
    double yj = y[i];
    double zj = z[i];
    double rj = sqrt(xj*xj + yj*yj);

    // Convert to cell units
    double r_cell = invdr*(rj - rmin) - 0.5;
    double z_cell = invdz*(zj - zmin) - 0.5;

    int ir_upper = (int)ceil(r_cell);
    int iz_upper = (int)ceil(z_cell);

    // Boundary conditions (aperiodic r, periodic z)
    if (ir_upper > Nr){
        ir_upper = Nr;
    }
    if (iz_upper < 0){
        iz_upper += Nz;
    } else if (iz_upper > Nz-1){
        iz_upper -= Nz;
    }

    sorted_idx[i] = i;
    cell_idx[i] = ir_upper + iz_upper * (Nr+1);
}


extern "C" __global__
void incl_prefix_sum(
    int* cell_idx,
    int* prefix_sum,
    int Ntot
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= Ntot - 1) return;

    int ci = cell_idx[i];
    int ci_next = cell_idx[i+1];

    while (ci < ci_next){
        prefix_sum[ci] = i + 1;
        ci += 1;
    }
}

extern "C" __global__
void prefill_prefix_sum(
    const int* cell_idx,
    int* prefix_sum,
    const int Ntot,
    const int Ncells
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= Ncells) return;

    if (Ntot > 0){
        int first = cell_idx[0];
        int last  = cell_idx[Ntot-1];

        if (i < first){
            prefix_sum[i] = 0;
        }
        else if (i >= last){
            prefix_sum[i] = Ntot;
        }
    }
    else{
        prefix_sum[i] = 0;
    }
}

extern "C" __global__
void write_sorting_buffer(
    long long* sorted_idx,
    const double* val,
    double* buf,
    const int Ntot
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= Ntot) return;

    buf[i] = val[ sorted_idx[i] ];
}
'''

get_cell_idx_per_particle_c = cp.RawKernel( src, "get_cell_idx_per_particle")
incl_prefix_sum_c = cp.RawKernel(src, "incl_prefix_sum")
prefill_prefix_sum_c = cp.RawKernel(src, "prefill_prefix_sum")
write_sorting_buffer_c = cp.RawKernel(src, "write_sorting_buffer")


@catch_gpu_memory_error
def sort_particles_per_cell(cell_idx, sorted_idx):
    """
    Sort the cell index of the particles and
    modify the sorted index array accordingly.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle

    sorted_idx : 1darray of integers
        Represents the original index of the
        particle before the sorting.
    """
    Ntot = cell_idx.shape[0]
    if Ntot > 0:
        if type(cell_idx) == np.ndarray or  type(sorted_idx) == np.ndarray:
            raise ValueError("Unexpected CPU array")
        d_cell_idx = cell_idx
        d_sorted_idx = sorted_idx
        # `thrust.argsort` will simultaneously:
        # - find the indices `sorted_idx` that sort the initial array cell_idx
        # - sort `cell_idx` in place
        thrust.argsort( dtype=d_cell_idx.dtype,
                        idx_start=d_sorted_idx.data.ptr,
                        data_start=d_cell_idx.data.ptr,
                        keys_start=0,
                        shape=d_cell_idx.shape )
        # As part of `thrust.argsort`, `cupy` will allocate temporarily
        # arrays in its memory pool. For performance reasons, this
        # memory is not automatically released, after `argsort`
        # Here we force `cupy` to release the memory.

