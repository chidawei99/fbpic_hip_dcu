# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the field gathering methods linear and cubic order shapes
on the GPU using CUDA.
"""
import cupy as cp

# Import inline functions


# -----------------------
# Field gathering linear
# -----------------------

src = r'''
#include <cupy/complex.cuh>

extern "C" __device__ void add_linear_gather_for_mode(
    const int m,
    double &Fr, double &Ft, double &Fz,
    const complex<double> exptheta_m,
    const complex<double>* __restrict__ Fr_grid,
    const complex<double>* __restrict__ Ft_grid,
    const complex<double>* __restrict__ Fz_grid,
    const int iz_lower, const int iz_upper,
    const int ir_lower, const int ir_upper,
    const double S_ll, const double S_lu, const double S_lg,
    const double S_ul, const double S_uu, const double S_ug,
    const int Nr, const int Nz )
{
    complex<double> Fr_m(0.0,0.0);
    complex<double> Ft_m(0.0,0.0);
    complex<double> Fz_m(0.0,0.0);

    // Lower z, Lower r
    Fr_m += S_ll * Fr_grid[iz_lower*Nr + ir_lower];
    Ft_m += S_ll * Ft_grid[iz_lower*Nr + ir_lower];
    Fz_m += S_ll * Fz_grid[iz_lower*Nr + ir_lower];
    // Lower z, Upper r
    Fr_m += S_lu * Fr_grid[iz_lower*Nr + ir_upper];
    Ft_m += S_lu * Ft_grid[iz_lower*Nr + ir_upper];
    Fz_m += S_lu * Fz_grid[iz_lower*Nr + ir_upper];
    // Upper z, Lower r
    Fr_m += S_ul * Fr_grid[iz_upper*Nr + ir_lower];
    Ft_m += S_ul * Ft_grid[iz_upper*Nr + ir_lower];
    Fz_m += S_ul * Fz_grid[iz_upper*Nr + ir_lower];
    // Upper z, Upper r
    Fr_m += S_uu * Fr_grid[iz_upper*Nr + ir_upper];
    Ft_m += S_uu * Ft_grid[iz_upper*Nr + ir_upper];
    Fz_m += S_uu * Fz_grid[iz_upper*Nr + ir_upper];

    // Guard cell contribution
    if (ir_lower==0 && ir_upper==0) {
        double flip_factor = (m%2==0 ? 1.0 : -1.0);
        Fr_m += -flip_factor * S_lg * Fr_grid[iz_lower*Nr + 0];
        Ft_m += -flip_factor * S_lg * Ft_grid[iz_lower*Nr + 0];
        Fz_m +=  flip_factor * S_lg * Fz_grid[iz_lower*Nr + 0];
        Fr_m += -flip_factor * S_ug * Fr_grid[iz_upper*Nr + 0];
        Ft_m += -flip_factor * S_ug * Ft_grid[iz_upper*Nr + 0];
        Fz_m +=  flip_factor * S_ug * Fz_grid[iz_upper*Nr + 0];
    }

    double factor = (m==0) ? 1.0 : 2.0;
    Fr += factor * (Fr_m*exptheta_m).real();
    Ft += factor * (Ft_m*exptheta_m).real();
    Fz += factor * (Fz_m*exptheta_m).real();
}

extern "C" __device__ void add_cubic_gather_for_mode(
    int m,
    double &Fr, double &Ft, double &Fz,
    const complex<double> &exptheta_m,
    const complex<double>* Er_grid,
    const complex<double>* Et_grid,
    const complex<double>* Ez_grid,
    int ir_lowest, int iz_lowest,
    const double* Sr, const double* Sz,
    int Nr, int Nz )
{
    complex<double> Fr_m = complex<double>(0.0, 0.0);
    complex<double> Ft_m = complex<double>(0.0, 0.0);
    complex<double> Fz_m = complex<double>(0.0, 0.0);

    // Loop over 4x4 stencil
    for (int iz=0; iz<4; iz++) {
        int iz_idx = iz_lowest + iz;
        if (iz_idx < 0) iz_idx += Nz;
        if (iz_idx >= Nz) iz_idx -= Nz;
        double Sz_val = Sz[iz];
        for (int ir=0; ir<4; ir++) {
            int ir_idx = ir_lowest + ir;
            if (ir_idx < 0) ir_idx = 0;
            if (ir_idx >= Nr) ir_idx = Nr-1;
            double S = Sz_val * Sr[ir];
            int idx = iz_idx*Nr + ir_idx;
            Fr_m += S * Er_grid[idx];
            Ft_m += S * Et_grid[idx];
            Fz_m += S * Ez_grid[idx];
        }
    }

    double factor = (m==0) ? 1.0 : 2.0;
    Fr += factor * (Fr_m * exptheta_m).real();
    Ft += factor * (Ft_m * exptheta_m).real();
    Fz += factor * (Fz_m * exptheta_m).real();
}

extern "C" __global__ void gather_field_gpu_linear(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double rmax_gather,
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    const complex<double>* __restrict__ Er_m0,
    const complex<double>* __restrict__ Et_m0,
    const complex<double>* __restrict__ Ez_m0,
    const complex<double>* __restrict__ Er_m1,
    const complex<double>* __restrict__ Et_m1,
    const complex<double>* __restrict__ Ez_m1,
    const complex<double>* __restrict__ Br_m0,
    const complex<double>* __restrict__ Bt_m0,
    const complex<double>* __restrict__ Bz_m0,
    const complex<double>* __restrict__ Br_m1,
    const complex<double>* __restrict__ Bt_m1,
    const complex<double>* __restrict__ Bz_m1,
    double* __restrict__ Ex,
    double* __restrict__ Ey,
    double* __restrict__ Ez,
    double* __restrict__ Bx,
    double* __restrict__ By,
    double* __restrict__ Bz,
    const int Np )
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= Np) return;

    double xj = x[i];
    double yj = y[i];
    double zj = z[i];

    double rj = sqrt(xj*xj + yj*yj);
    double cosv, sinv;
    if (rj != 0.0){
        cosv = xj / rj;
        sinv = yj / rj;
    } else {
        cosv = 1.0;
        sinv = 0.0;
    }

    complex<double> exptheta_m0(1.0,0.0);
    complex<double> exptheta_m1(cosv, -sinv);

    double r_cell = invdr*(rj - rmin) - 0.5;
    double z_cell = invdz*(zj - zmin) - 0.5;

    int ir_lower = floor(r_cell);
    int ir_upper = ir_lower + 1;
    int iz_lower = floor(z_cell);
    int iz_upper = iz_lower + 1;

    double Sr_lower = ir_upper - r_cell;
    double Sr_upper = r_cell - ir_lower;
    double Sz_lower = iz_upper - z_cell;
    double Sz_upper = z_cell - iz_lower;
    double Sr_guard = 0.0;

    if (ir_lower < 0){
        Sr_guard = Sr_lower;
        Sr_lower = 0.0;
        ir_lower = 0;
    }
    if (ir_lower > Nr-1) ir_lower = Nr-1;
    if (ir_upper > Nr-1) ir_upper = Nr-1;

    if (iz_lower < 0) iz_lower += Nz;
    if (iz_upper < 0) iz_upper += Nz;
    if (iz_lower > Nz-1) iz_lower -= Nz;
    if (iz_upper > Nz-1) iz_upper -= Nz;

    double S_ll = Sz_lower*Sr_lower;
    double S_lu = Sz_lower*Sr_upper;
    double S_ul = Sz_upper*Sr_lower;
    double S_uu = Sz_upper*Sr_upper;
    double S_lg = Sz_lower*Sr_guard;
    double S_ug = Sz_upper*Sr_guard;

    double Fr=0.0, Ft=0.0, Fzv=0.0;

    if (rj < rmax_gather){
        add_linear_gather_for_mode(0, Fr,Ft,Fzv, exptheta_m0,
            Er_m0, Et_m0, Ez_m0,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug, Nr, Nz);
        add_linear_gather_for_mode(1, Fr,Ft,Fzv, exptheta_m1,
            Er_m1, Et_m1, Ez_m1,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug, Nr, Nz);
    }

    Ex[i] = cosv*Fr - sinv*Ft;
    Ey[i] = sinv*Fr + cosv*Ft;
    Ez[i] = Fzv;

    Fr=0.0; Ft=0.0; Fzv=0.0;
    if (rj < rmax_gather){
        add_linear_gather_for_mode(0, Fr,Ft,Fzv, exptheta_m0,
            Br_m0, Bt_m0, Bz_m0,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug, Nr, Nz);
        add_linear_gather_for_mode(1, Fr,Ft,Fzv, exptheta_m1,
            Br_m1, Bt_m1, Bz_m1,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug, Nr, Nz);
    }

    Bx[i] = cosv*Fr - sinv*Ft;
    By[i] = sinv*Fr + cosv*Ft;
    Bz[i] = Fzv;
}

extern "C" __global__ void gather_field_gpu_cubic(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double rmax_gather,
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    const complex<double>* __restrict__ Er_m0,
    const complex<double>* __restrict__ Et_m0,
    const complex<double>* __restrict__ Ez_m0,
    const complex<double>* __restrict__ Er_m1,
    const complex<double>* __restrict__ Et_m1,
    const complex<double>* __restrict__ Ez_m1,
    const complex<double>* __restrict__ Br_m0,
    const complex<double>* __restrict__ Bt_m0,
    const complex<double>* __restrict__ Bz_m0,
    const complex<double>* __restrict__ Br_m1,
    const complex<double>* __restrict__ Bt_m1,
    const complex<double>* __restrict__ Bz_m1,
    double* __restrict__ Ex,
    double* __restrict__ Ey,
    double* __restrict__ Ez,
    double* __restrict__ Bx,
    double* __restrict__ By,
    double* __restrict__ Bz,
    const int Np )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Np) return;

    double xj = x[i];
    double yj = y[i];
    double zj = z[i];

    double rj = sqrt(xj*xj + yj*yj);
    double cos_t, sin_t;
    if (rj != 0.0) {
        double invr = 1.0/rj;
        cos_t = xj*invr;
        sin_t = yj*invr;
    } else {
        cos_t = 1.0;
        sin_t = 0.0;
    }
    complex<double> exptheta_m0 = complex<double>(1.0,0.0);
    complex<double> exptheta_m1 = complex<double>(cos_t, -sin_t);

    // Shape factors
    double r_cell = invdr*(rj - rmin) - 0.5;
    double z_cell = invdz*(zj - zmin) - 0.5;

    int ir_lowest = (int)floor(r_cell) - 1;
    double r_local = r_cell - ir_lowest;
    double Sr[4];
    Sr[0] = -1.0/6.0 * pow(r_local-2.0,3);
    Sr[1] =  1.0/6.0 * (3.0*pow(r_local-1.0,3) - 6.0*pow(r_local-1.0,2) + 4.0);
    Sr[2] =  1.0/6.0 * (3.0*pow(2.0-r_local,3) - 6.0*pow(2.0-r_local,2) + 4.0);
    Sr[3] = -1.0/6.0 * pow(1.0-r_local,3);

    int iz_lowest = (int)floor(z_cell) - 1;
    double z_local = z_cell - iz_lowest;
    double Sz[4];
    Sz[0] = -1.0/6.0 * pow(z_local-2.0,3);
    Sz[1] =  1.0/6.0 * (3.0*pow(z_local-1.0,3) - 6.0*pow(z_local-1.0,2) + 4.0);
    Sz[2] =  1.0/6.0 * (3.0*pow(2.0-z_local,3) - 6.0*pow(2.0-z_local,2) + 4.0);
    Sz[3] = -1.0/6.0 * pow(1.0-z_local,3);

    double Fr=0.0, Ft=0.0, Fz=0.0;
    if (rj < rmax_gather) {
        add_cubic_gather_for_mode(0, Fr, Ft, Fz, exptheta_m0,
            Er_m0, Et_m0, Ez_m0, ir_lowest, iz_lowest, Sr, Sz, Nr, Nz);
        add_cubic_gather_for_mode(1, Fr, Ft, Fz, exptheta_m1,
            Er_m1, Et_m1, Ez_m1, ir_lowest, iz_lowest, Sr, Sz, Nr, Nz);
    }
    Ex[i] = cos_t*Fr - sin_t*Ft;
    Ey[i] = sin_t*Fr + cos_t*Ft;
    Ez[i] = Fz;

    Fr=Ft=Fz=0.0;
    if (rj < rmax_gather) {
        add_cubic_gather_for_mode(0, Fr, Ft, Fz, exptheta_m0,
            Br_m0, Bt_m0, Bz_m0, ir_lowest, iz_lowest, Sr, Sz, Nr, Nz);
        add_cubic_gather_for_mode(1, Fr, Ft, Fz, exptheta_m1,
            Br_m1, Bt_m1, Bz_m1, ir_lowest, iz_lowest, Sr, Sz, Nr, Nz);
    }
    Bx[i] = cos_t*Fr - sin_t*Ft;
    By[i] = sin_t*Fr + cos_t*Ft;
    Bz[i] = Fz;
}

'''
gather_field_gpu_linear_c = cp.RawKernel(src, 'gather_field_gpu_linear')
gather_field_gpu_cubic_c = cp.RawKernel(src, 'gather_field_gpu_cubic')

