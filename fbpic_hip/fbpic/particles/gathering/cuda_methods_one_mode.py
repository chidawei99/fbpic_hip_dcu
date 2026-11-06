# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the field gathering methods linear and cubic order shapes
on the GPU using CUDA, for one azimuthal mode at a time
"""
# Import inline functions

# Compile the inline functions for GPU


import cupy as cp

src = r'''
#include <cupy/complex.cuh>


// -------------------------------
// 设备函数：add_linear_gather_for_mode
// -------------------------------
__device__ inline void add_linear_gather_for_mode(
    const int m,
    double &Fr, double &Ft, double &Fz,
    const complex<double> &exptheta_m,
    const complex<double>* __restrict__ Fr_grid,
    const complex<double>* __restrict__ Ft_grid,
    const complex<double>* __restrict__ Fz_grid,
    const int iz_lower, const int iz_upper,
    const int ir_lower, const int ir_upper,
    const double S_ll, const double S_lu, const double S_lg,
    const double S_ul, const double S_uu, const double S_ug,
    const int Nr
){
    complex<double> Fr_m(0.0, 0.0);
    complex<double> Ft_m(0.0, 0.0);
    complex<double> Fz_m(0.0, 0.0);
    int idx;

    // 四个 cell 累加
    idx = iz_lower * Nr + ir_lower;
    Fr_m += S_ll * Fr_grid[idx];
    Ft_m += S_ll * Ft_grid[idx];
    Fz_m += S_ll * Fz_grid[idx];

    idx = iz_lower * Nr + ir_upper;
    Fr_m += S_lu * Fr_grid[idx];
    Ft_m += S_lu * Ft_grid[idx];
    Fz_m += S_lu * Fz_grid[idx];

    idx = iz_upper * Nr + ir_lower;
    Fr_m += S_ul * Fr_grid[idx];
    Ft_m += S_ul * Ft_grid[idx];
    Fz_m += S_ul * Fz_grid[idx];

    idx = iz_upper * Nr + ir_upper;
    Fr_m += S_uu * Fr_grid[idx];
    Ft_m += S_uu * Ft_grid[idx];
    Fz_m += S_uu * Fz_grid[idx];

    // Guard cells
    if (ir_lower == 0 && ir_upper == 0) {
        double flip_factor = (m % 2 == 0 ? 1.0 : -1.0);

        idx = iz_lower * Nr + 0;
        Fr_m += -flip_factor * S_lg * Fr_grid[idx];
        Ft_m += -flip_factor * S_lg * Ft_grid[idx];
        Fz_m +=  flip_factor * S_lg * Fz_grid[idx];

        idx = iz_upper * Nr + 0;
        Fr_m += -flip_factor * S_ug * Fr_grid[idx];
        Ft_m += -flip_factor * S_ug * Ft_grid[idx];
        Fz_m +=  flip_factor * S_ug * Fz_grid[idx];
    }

    // mode 权重
    double factor = (m == 0) ? 1.0 : 2.0;

    Fr += factor * (Fr_m * exptheta_m).real();
    Ft += factor * (Ft_m * exptheta_m).real();
    Fz += factor * (Fz_m * exptheta_m).real();
}

// -------------------------------
// 设备函数：add_cubic_gather_for_mode
// -------------------------------
__device__ inline void add_cubic_gather_for_mode(
    const int m,
    double &Fr, double &Ft, double &Fz,
    const complex<double> &exptheta_m,
    const complex<double>* __restrict__ Fr_grid,
    const complex<double>* __restrict__ Ft_grid,
    const complex<double>* __restrict__ Fz_grid,
    int ir_lowest, int iz_lowest,
    const double* __restrict__ Sr_arr,
    const double* __restrict__ Sz_arr,
    const int Nr, const int Nz
){
    complex<double> Fr_m(0.0, 0.0);
    complex<double> Ft_m(0.0, 0.0);
    complex<double> Fz_m(0.0, 0.0);

    for (int index_r=0; index_r<4; index_r++) {
        int ir = ir_lowest + index_r;
        double Sr_long = Sr_arr[index_r];
        double Sr_perp = Sr_arr[index_r];

        // axis correction
        if (ir < 0) {
            Sr_long *= ((m % 2 == 0) ? 1.0 : -1.0);
            Sr_perp *= -((m % 2 == 0) ? 1.0 : -1.0);
            ir = -ir - 1; // reflect index
        } else if (ir > Nr-1) {
            ir = Nr-1;
        }

        for (int index_z=0; index_z<4; index_z++) {
            int iz = iz_lowest + index_z;
            double Sz = Sz_arr[index_z];

            // periodic z
            if (iz < 0) iz += Nz;
            else if (iz > Nz-1) iz -= Nz;

            int idx = iz * Nr + ir;
            Fr_m += Sz * Sr_perp * Fr_grid[idx];
            Ft_m += Sz * Sr_perp * Ft_grid[idx];
            Fz_m += Sz * Sr_long * Fz_grid[idx];
        }
    }

    double factor = (m == 0) ? 1.0 : 2.0;
    Fr += factor * (Fr_m * exptheta_m).real();
    Ft += factor * (Ft_m * exptheta_m).real();
    Fz += factor * (Fz_m * exptheta_m).real();
}

extern "C" __global__
void erase_eb_cuda(
    double* __restrict__ Ex,
    double* __restrict__ Ey,
    double* __restrict__ Ez,
    double* __restrict__ Bx,
    double* __restrict__ By,
    double* __restrict__ Bz,
    const int Ntot
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ntot) {
        Ex[i] = 0.0;
        Ey[i] = 0.0;
        Ez[i] = 0.0;
        Bx[i] = 0.0;
        By[i] = 0.0;
        Bz[i] = 0.0;
    }
}
// -------------------------------
// 主核函数：gather_field_gpu_linear_one_mode
// -------------------------------
extern "C" __global__
void gather_field_gpu_linear_one_mode(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double rmax_gather,
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    const complex<double>* __restrict__ Er_m,
    const complex<double>* __restrict__ Et_m,
    const complex<double>* __restrict__ Ez_m,
    const complex<double>* __restrict__ Br_m,
    const complex<double>* __restrict__ Bt_m,
    const complex<double>* __restrict__ Bz_m,
    const int m,
    double* __restrict__ Ex,
    double* __restrict__ Ey,
    double* __restrict__ Ez,
    double* __restrict__ Bx,
    double* __restrict__ By,
    double* __restrict__ Bz,
    const int Npart   // 粒子总数
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Npart) return;

    // -------------------------------
    // 位置和极坐标
    // -------------------------------
    double xj = x[i];
    double yj = y[i];
    double zj = z[i];

    double rj = sqrt(xj*xj + yj*yj);
    double invr, cosv, sinv;
    if (rj != 0.0) {
        invr = 1.0 / rj;
        cosv = xj * invr;
        sinv = yj * invr;
    } else {
        cosv = 1.0;
        sinv = 0.0;
    }

    // azimuthal factor exp(-i m θ)
    complex<double> exptheta_m(1.0, 0.0);
    for (int k=0; k<m; k++) {
        exptheta_m *= complex<double>(cosv, -sinv);
    }

    // cell 坐标
    double r_cell = invdr * (rj - rmin) - 0.5;
    double z_cell = invdz * (zj - zmin) - 0.5;

    if (rj < rmax_gather) {
        int ir_lower = (int)floor(r_cell);
        int ir_upper = ir_lower + 1;
        int iz_lower = (int)floor(z_cell);
        int iz_upper = iz_lower + 1;

        double Sr_lower = ir_upper - r_cell;
        double Sr_upper = r_cell - ir_lower;
        double Sz_lower = iz_upper - z_cell;
        double Sz_upper = z_cell - iz_lower;
        double Sr_guard = 0.0;

        // r 边界
        if (ir_lower < 0) {
            Sr_guard = Sr_lower;
            Sr_lower = 0.0;
            ir_lower = 0;
        }
        if (ir_lower > Nr-1) ir_lower = Nr-1;
        if (ir_upper > Nr-1) ir_upper = Nr-1;

        // z 周期边界
        if (iz_lower < 0) iz_lower += Nz;
        if (iz_upper < 0) iz_upper += Nz;
        if (iz_lower > Nz-1) iz_lower -= Nz;
        if (iz_upper > Nz-1) iz_upper -= Nz;

        // shape 函数
        double S_ll = Sz_lower*Sr_lower;
        double S_lu = Sz_lower*Sr_upper;
        double S_ul = Sz_upper*Sr_lower;
        double S_uu = Sz_upper*Sr_upper;
        double S_lg = Sz_lower*Sr_guard;
        double S_ug = Sz_upper*Sr_guard;

        // -------------------------------
        // Gather E field
        // -------------------------------
        double Fr=0.0, Ft=0.0, Fz_acc=0.0;
        add_linear_gather_for_mode(
            m, Fr, Ft, Fz_acc, exptheta_m,
            Er_m, Et_m, Ez_m,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug, Nr
        );
        Ex[i] += cosv*Fr - sinv*Ft;
        Ey[i] += sinv*Fr + cosv*Ft;
        Ez[i] += Fz_acc;

        // -------------------------------
        // Gather B field
        // -------------------------------
        Fr=0.0; Ft=0.0; Fz_acc=0.0;
        add_linear_gather_for_mode(
            m, Fr, Ft, Fz_acc, exptheta_m,
            Br_m, Bt_m, Bz_m,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug, Nr
        );
        Bx[i] += cosv*Fr - sinv*Ft;
        By[i] += sinv*Fr + cosv*Ft;
        Bz[i] += Fz_acc;
    }
}

// -------------------------------
// 主核函数：gather_field_gpu_cubic_one_mode
// -------------------------------
extern "C" __global__
void gather_field_gpu_cubic_one_mode(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double rmax_gather,
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    const complex<double>* __restrict__ Er_m,
    const complex<double>* __restrict__ Et_m,
    const complex<double>* __restrict__ Ez_m,
    const complex<double>* __restrict__ Br_m,
    const complex<double>* __restrict__ Bt_m,
    const complex<double>* __restrict__ Bz_m,
    const int m,
    double* __restrict__ Ex,
    double* __restrict__ Ey,
    double* __restrict__ Ez,
    double* __restrict__ Bx,
    double* __restrict__ By,
    double* __restrict__ Bz,
    const int Npart
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Npart) return;

    // -------------------------------
    // particle pos & cylindrical conversion
    // -------------------------------
    double xj = x[i];
    double yj = y[i];
    double zj = z[i];

    double rj = sqrt(xj*xj + yj*yj);
    double cosv, sinv;
    if (rj != 0.0) {
        double invr = 1.0/rj;
        cosv = xj * invr;
        sinv = yj * invr;
    } else {
        cosv = 1.0;
        sinv = 0.0;
    }

    // exp(-i m θ)
    complex<double> exptheta_m(1.0, 0.0);
    for (int k=0; k<m; k++) {
        exptheta_m *= complex<double>(cosv, -sinv);
    }

    double r_cell = invdr*(rj - rmin) - 0.5;
    double z_cell = invdz*(zj - zmin) - 0.5;

    if (rj < rmax_gather) {
        // -------------------------------
        // cubic weights
        // -------------------------------
        int ir_lowest = (int)floor(r_cell) - 1;
        double r_local = r_cell - ir_lowest;
        double Sr[4];
        Sr[0] = -1.0/6.0 * pow(r_local-2.0, 3);
        Sr[1] =  1.0/6.0 * (3.0*pow(r_local-1.0,3) - 6.0*pow(r_local-1.0,2) + 4.0);
        Sr[2] =  1.0/6.0 * (3.0*pow(2.0-r_local,3) - 6.0*pow(2.0-r_local,2) + 4.0);
        Sr[3] = -1.0/6.0 * pow(1.0-r_local, 3);

        int iz_lowest = (int)floor(z_cell) - 1;
        double z_local = z_cell - iz_lowest;
        double Sz[4];
        Sz[0] = -1.0/6.0 * pow(z_local-2.0, 3);
        Sz[1] =  1.0/6.0 * (3.0*pow(z_local-1.0,3) - 6.0*pow(z_local-1.0,2) + 4.0);
        Sz[2] =  1.0/6.0 * (3.0*pow(2.0-z_local,3) - 6.0*pow(2.0-z_local,2) + 4.0);
        Sz[3] = -1.0/6.0 * pow(1.0-z_local, 3);

        // -------------------------------
        // Gather E
        // -------------------------------
        double Fr=0.0, Ft=0.0, Fz_acc=0.0;
        add_cubic_gather_for_mode(
            m, Fr, Ft, Fz_acc, exptheta_m,
            Er_m, Et_m, Ez_m,
            ir_lowest, iz_lowest, Sr, Sz, Nr, Nz
        );
        Ex[i] += cosv*Fr - sinv*Ft;
        Ey[i] += sinv*Fr + cosv*Ft;
        Ez[i] += Fz_acc;

        // -------------------------------
        // Gather B
        // -------------------------------
        Fr=0.0; Ft=0.0; Fz_acc=0.0;
        add_cubic_gather_for_mode(
            m, Fr, Ft, Fz_acc, exptheta_m,
            Br_m, Bt_m, Bz_m,
            ir_lowest, iz_lowest, Sr, Sz, Nr, Nz
        );
        Bx[i] += cosv*Fr - sinv*Ft;
        By[i] += sinv*Fr + cosv*Ft;
        Bz[i] += Fz_acc;
    }
}

'''

erase_eb_cuda_c = cp.RawKernel(src, 'erase_eb_cuda')
gather_field_gpu_linear_one_mode_c = cp.RawKernel(src, 'gather_field_gpu_linear_one_mode')
gather_field_gpu_cubic_one_mode_c = cp.RawKernel(src, 'gather_field_gpu_cubic_one_mode')

