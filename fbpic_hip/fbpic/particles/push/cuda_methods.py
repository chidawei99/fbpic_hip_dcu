# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle push methods on the GPU using CUDA.
"""
from scipy.constants import c, e
# Import inline function


import cupy as cp

src = r'''


__device__ void push_p_vay(
    double ux_i, double uy_i, double uz_i, double inv_gamma_i,
    double Ex, double Ey, double Ez,
    double Bx, double By, double Bz,
    double econst, double bconst,
    double* ux_f, double* uy_f, double* uz_f, double* inv_gamma_f
){
    // Magnetic rotation vector
    double taux = bconst * Bx;
    double tauy = bconst * By;
    double tauz = bconst * Bz;
    double tau2 = taux*taux + tauy*tauy + tauz*tauz;

    // Momenta at half timestep
    double uxp = ux_i + econst*Ex + inv_gamma_i*( uy_i*tauz - uz_i*tauy );
    double uyp = uy_i + econst*Ey + inv_gamma_i*( uz_i*taux - ux_i*tauz );
    double uzp = uz_i + econst*Ez + inv_gamma_i*( ux_i*tauy - uy_i*taux );
    double sigma = 1.0 + uxp*uxp + uyp*uyp + uzp*uzp - tau2;
    double utau = uxp*taux + uyp*tauy + uzp*tauz;

    // New 1/gamma
    double inv_gamma_tmp = sqrt( 2.0 / ( sigma + sqrt( sigma*sigma + 4.0*(tau2 + utau*utau) ) ) );

    // Intermediate terms
    double tx = inv_gamma_tmp * taux;
    double ty = inv_gamma_tmp * tauy;
    double tz = inv_gamma_tmp * tauz;
    double ut = inv_gamma_tmp * utau;
    double s = 1.0 / ( 1.0 + tau2*inv_gamma_tmp*inv_gamma_tmp );

    // New u
    *ux_f = s * ( uxp + tx*ut + uyp*tz - uzp*ty );
    *uy_f = s * ( uyp + ty*ut + uzp*tx - uxp*tz );
    *uz_f = s * ( uzp + tz*ut + uxp*ty - uyp*tx );
    *inv_gamma_f = inv_gamma_tmp;
}

extern "C" __global__
void push_x_gpu(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ inv_gamma,
    const double dt,
    const double x_push,
    const double y_push,
    const double z_push,
    const int Np,
    const double c
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Np) return;

    double inv_g = inv_gamma[i];
    double cdt = c * dt;

    x[i] += cdt * x_push * inv_g * ux[i];
    y[i] += cdt * y_push * inv_g * uy[i];
    z[i] += cdt * z_push * inv_g * uz[i];
}

extern "C" __global__
void push_p_gpu(
    double* __restrict__ ux,
    double* __restrict__ uy,
    double* __restrict__ uz,
    double* __restrict__ inv_gamma,
    const double* __restrict__ Ex,
    const double* __restrict__ Ey,
    const double* __restrict__ Ez,
    const double* __restrict__ Bx,
    const double* __restrict__ By,
    const double* __restrict__ Bz,
    const double q, const double m,
    const double dt,
    const int Ntot,
    const double c
){
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= Ntot) return;

    double econst = q * dt / (m * c);
    double bconst = 0.5 * q * dt / m;

    double ux_f, uy_f, uz_f, inv_gamma_f;

    push_p_vay(
        ux[ip], uy[ip], uz[ip], inv_gamma[ip],
        Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
        econst, bconst,
        &ux_f, &uy_f, &uz_f, &inv_gamma_f
    );

    ux[ip] = ux_f;
    uy[ip] = uy_f;
    uz[ip] = uz_f;
    inv_gamma[ip] = inv_gamma_f;
}

extern "C" __global__
void push_p_after_plane_gpu(
    const double* __restrict__ z,
    const double z_plane,
    double* __restrict__ ux,
    double* __restrict__ uy,
    double* __restrict__ uz,
    double* __restrict__ inv_gamma,
    const double* __restrict__ Ex,
    const double* __restrict__ Ey,
    const double* __restrict__ Ez,
    const double* __restrict__ Bx,
    const double* __restrict__ By,
    const double* __restrict__ Bz,
    const double q, const double m,
    const int Ntot, const double dt,
    const double c
){
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= Ntot) return;

    if (z[ip] > z_plane) {
        double econst = q * dt / (m * c);
        double bconst = 0.5 * q * dt / m;

        double ux_f, uy_f, uz_f, inv_gamma_f;

        push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
            econst, bconst,
            &ux_f, &uy_f, &uz_f, &inv_gamma_f
        );

        ux[ip] = ux_f;
        uy[ip] = uy_f;
        uz[ip] = uz_f;
        inv_gamma[ip] = inv_gamma_f;
    }
}

extern "C" __global__
void push_p_ioniz_gpu(
    double* __restrict__ ux,
    double* __restrict__ uy,
    double* __restrict__ uz,
    double* __restrict__ inv_gamma,
    const double* __restrict__ Ex,
    const double* __restrict__ Ey,
    const double* __restrict__ Ez,
    const double* __restrict__ Bx,
    const double* __restrict__ By,
    const double* __restrict__ Bz,
    const double m,
    const int Ntot,
    const double dt,
    const int* __restrict__ ionization_level,
    const double e,
    const double c
){
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= Ntot) return;

    int Z = ionization_level[ip];
    if (Z != 0) {
        double econst = Z * e * dt / (m * c);
        double bconst = 0.5 * Z * e * dt / m;

        double ux_f, uy_f, uz_f, inv_gamma_f;
        push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
            econst, bconst,
            &ux_f, &uy_f, &uz_f, &inv_gamma_f
        );

        ux[ip] = ux_f;
        uy[ip] = uy_f;
        uz[ip] = uz_f;
        inv_gamma[ip] = inv_gamma_f;
    }
}
'''
push_x_gpu_c = cp.RawKernel(src, 'push_x_gpu')
push_p_gpu_c = cp.RawKernel(src, 'push_p_gpu')
push_p_after_plane_gpu_c = cp.RawKernel(src, 'push_p_after_plane_gpu')
push_p_ioniz_gpu_c = cp.RawKernel(src, 'push_p_ioniz_gpu')

