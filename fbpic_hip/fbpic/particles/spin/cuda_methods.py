# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin
# Scientific supervision: Matteo Tamburini
# Code optimization: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This file is for the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It carries out the spin-push related methods for all particles
on GPU using CUDA.
"""
from scipy.constants import e, c
# Import inline function


# Compile the inline function for GPU


import cupy as cp

src = r'''
__device__ void push_s_BMT(
    double sx_i, double sy_i, double sz_i,
    double ux_i, double uy_i, double uz_i,
    double ux_f, double uy_f, double uz_f,
    double Ex, double Ey, double Ez,
    double Bx, double By, double Bz,
    double tauconst, double anom,
    double c,
    double* sx_f, double* sy_f, double* sz_f
){
    double ux = 0.5 * (ux_i + ux_f);
    double uy = 0.5 * (uy_i + uy_f);
    double uz = 0.5 * (uz_i + uz_f);

    double inv_gamma = 1.0 / sqrt( 1.0 + ux*ux + uy*uy + uz*uz );
    double inv_1pgamma = inv_gamma / (1.0 + inv_gamma);
    double uB = ux*Bx + uy*By + uz*Bz;

    double taux = tauconst * (
        (anom + inv_gamma) * Bx
        - (anom + inv_1pgamma) * (uy*Ez - uz*Ey) * inv_gamma / c
        - anom * inv_gamma * inv_1pgamma * uB * ux
    );
    double tauy = tauconst * (
        (anom + inv_gamma) * By
        - (anom + inv_1pgamma) * (uz*Ex - ux*Ez) * inv_gamma / c
        - anom * inv_gamma * inv_1pgamma * uB * uy
    );
    double tauz = tauconst * (
        (anom + inv_gamma) * Bz
        - (anom + inv_1pgamma) * (ux*Ey - uy*Ex) * inv_gamma / c
        - anom * inv_gamma * inv_1pgamma * uB * uz
    );

    double tau2 = taux*taux + tauy*tauy + tauz*tauz;

    double ups = 1.0 / (1.0 + tau2);
    double stau = sx_i*taux + sy_i*tauy + sz_i*tauz;

    *sx_f = ups * ( sx_i + 2.0*(sy_i*tauz - sz_i*tauy)
                  + stau*taux + (sz_i*taux - sx_i*tauz)*tauz
                  - (sx_i*tauy - sy_i*taux)*tauy );
    *sy_f = ups * ( sy_i + 2.0*(sz_i*taux - sx_i*tauz)
                  + stau*tauy + (sx_i*tauy - sy_i*taux)*taux
                  - (sy_i*tauz - sz_i*tauy)*tauz );
    *sz_f = ups * ( sz_i + 2.0*(sx_i*tauy - sy_i*taux)
                  + stau*tauz + (sy_i*tauz - sz_i*tauy)*tauy
                  - (sz_i*taux - sx_i*tauz)*taux );
}


// 设备函数：对应 Python 版本的 batch 内核
__device__ void copy_ionized_electron_spin_batch(
    int i_batch, int batch_size, int elec_old_Ntot, int ion_Ntot,
    long long* cumulative_n_ionized,  // shape [n_levels, N_batch]
    int n_batch, int i_level, short* ionized_from,
    bool store_electrons_per_level,
    double* __restrict__ elec_sx,
    double* __restrict__ elec_sy,
    double* __restrict__ elec_sz,
    const double* __restrict__ ion_sx,
    const double* __restrict__ ion_sy,
    const double* __restrict__ ion_sz,
    const double* __restrict__ rand_sx,
    const double* __restrict__ rand_sy,
    const double* __restrict__ rand_sz)
{
    // 索引到 cumulative_n_ionized[i_level, i_batch]
    int elec_index = elec_old_Ntot + cumulative_n_ionized[i_level * n_batch + i_batch];

    // 当前 batch 范围
    int start = i_batch * batch_size;
    int end   = min((i_batch + 1) * batch_size, ion_Ntot);

    for (int ion_index = start; ion_index < end; ion_index++){
        bool create_electron = false;

        if ((!store_electrons_per_level) && (ionized_from[ion_index] >= 0)){
            create_electron = true;
        }
        if (store_electrons_per_level && (ionized_from[ion_index] == i_level)){
            create_electron = true;
        }

        if (create_electron){
            if (ionized_from[ion_index] == 0){
                // 复制母离子自旋
                elec_sx[elec_index] = ion_sx[ion_index];
                elec_sy[elec_index] = ion_sy[ion_index];
                elec_sz[elec_index] = ion_sz[ion_index];
            }
            else{
                // 随机自旋
                int rand_index = elec_index - elec_old_Ntot;
                elec_sx[elec_index] = rand_sx[rand_index];
                elec_sy[elec_index] = rand_sy[rand_index];
                elec_sz[elec_index] = rand_sz[rand_index];
            }
            elec_index += 1;
        }
    }
}

extern "C" __global__
void push_s_gpu(
    double* __restrict__ sx,
    double* __restrict__ sy,
    double* __restrict__ sz,
    const double* __restrict__ ux_prev,
    const double* __restrict__ uy_prev,
    const double* __restrict__ uz_prev,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ Ex,
    const double* __restrict__ Ey,
    const double* __restrict__ Ez,
    const double* __restrict__ Bx,
    const double* __restrict__ By,
    const double* __restrict__ Bz,
    const double q,
    const double m,
    const int Ntot,
    const double dt,
    const double anom,
    const double c
){
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= Ntot) return;

    double tauconst = dt * q / (2.0 * m);

    double sx_f, sy_f, sz_f;
    push_s_BMT(
        sx[ip], sy[ip], sz[ip],
        ux_prev[ip], uy_prev[ip], uz_prev[ip],
        ux[ip], uy[ip], uz[ip],
        Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
        tauconst, anom, c,
        &sx_f, &sy_f, &sz_f
    );

    sx[ip] = sx_f;
    sy[ip] = sy_f;
    sz[ip] = sz_f;
}

extern "C" __global__
void push_s_ioniz_gpu(
    double* __restrict__ sx,
    double* __restrict__ sy,
    double* __restrict__ sz,
    const double* __restrict__ ux_prev,
    const double* __restrict__ uy_prev,
    const double* __restrict__ uz_prev,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ Ex,
    const double* __restrict__ Ey,
    const double* __restrict__ Ez,
    const double* __restrict__ Bx,
    const double* __restrict__ By,
    const double* __restrict__ Bz,
    const double m,
    const int Ntot,
    const double dt,
    const double anom,
    const int* __restrict__ ionization_level,
    const double e,
    const double c
){
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= Ntot) return;

    int Z = ionization_level[ip];
    if (Z != 0){
        double tauconst = Z * dt * e / (2.0 * m);

        double sx_f, sy_f, sz_f;
        push_s_BMT(
            sx[ip], sy[ip], sz[ip],
            ux_prev[ip], uy_prev[ip], uz_prev[ip],
            ux[ip], uy[ip], uz[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
            tauconst, anom, c,
            &sx_f, &sy_f, &sz_f
        );

        sx[ip] = sx_f;
        sy[ip] = sy_f;
        sz[ip] = sz_f;
    }
}

extern "C" __global__
void copy_ionized_electron_spin_cuda(
    int N_batch,
    int batch_size,
    int elec_old_Ntot,
    int ion_Ntot,
    bool store_electrons_per_level,
    long long*  cumulative_n_ionized,  // shape [n_levels, N_batch]
    int n_batch,  // cumulative_n_ionized 的第二维长度
    int i_level,
    short* __restrict__ ionized_from,
    double* __restrict__ elec_sx,
    double* __restrict__ elec_sy,
    double* __restrict__ elec_sz,
    const double* __restrict__ ion_sx,
    const double* __restrict__ ion_sy,
    const double* __restrict__ ion_sz,
    const double* __restrict__ rand_sx,
    const double* __restrict__ rand_sy,
    const double* __restrict__ rand_sz
){
    int i_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_batch < N_batch){
        copy_ionized_electron_spin_batch(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, n_batch, i_level, ionized_from,
            store_electrons_per_level,
            elec_sx, elec_sy, elec_sz,
            ion_sx, ion_sy, ion_sz,
            rand_sx, rand_sy, rand_sz
        );
    }
}
'''

push_s_gpu_c = cp.RawKernel(src, 'push_s_gpu')
push_s_ioniz_gpu_c = cp.RawKernel(src, 'push_s_ioniz_gpu')
copy_ionized_electron_spin_cuda_c = cp.RawKernel(src, 'copy_ionized_electron_spin_cuda')

