# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the GPU using CUDA, for one azimuthal mode only
"""
import math
from scipy.constants import c
import numpy as np

# JIT-compilation of particle shapes

# -------------------------------
# Field deposition - linear - rho
# -------------------------------

import cupy as cp

aux_src = r'''
#include <cupy/complex.cuh>

// longitudinal (z) shape factor
__device__ double Sz_linear(double cell_position, int index) {
    double s = ceil(cell_position) - cell_position;
    if (index == 1) {
        s = 1.0 - s;
    }
    return s;
}

// transversal (r) shape factor
__device__ double Sr_linear(double cell_position, int index, int flip, double beta_n) {
    int ir = int(ceil(cell_position)) - 1;
    double u = cell_position - ir; // between 0 and 1
    double s = (1.0 - u) + beta_n * (1.0 - u) * u;
    if (index == 1) {
        s = 1.0 - s;
    }
    if (index + ir < 0) {
        s *= flip;
    }
    return s;
}
// longitudinal (z) - cubic shape
__device__  double Sz_cubic(double cell_position, int index) {
    int iz = (int)ceil(cell_position) - 2;
    // u: position of the particle with respect to its left neighbor gridpoint
    double u = cell_position - iz - 1;  // u in [0,1]

    double s = 0.0;
    if (index == 0) {
        s = (1.0/6.0) * pow(1.0 - u, 3);
    }
    else if (index == 1) {
        s = (1.0/6.0) * (3.0*u*u*u - 6.0*u*u + 4.0);
    }
    else if (index == 2) {
        s = (1.0/6.0) * (3.0*pow(1.0 - u, 3) - 6.0*pow(1.0 - u, 2) + 4.0);
    }
    else if (index == 3) {
        s = (1.0/6.0) * (u*u*u);
    }
    return s;
}

// transversal (r) - cubic shape
__device__  double Sr_cubic(double cell_position, int index, int flip, double beta_n) {
    int ir = (int)ceil(cell_position) - 2;
    double u = cell_position - ir - 1;  // u in [0,1]

    double s = 0.0;
    if (index == 0) {
        s = (1.0/6.0) * pow(1.0 - u, 3);
    }
    else if (index == 1) {
        s = (1.0/6.0) * (3.0*u*u*u - 6.0*u*u + 4.0);
        s += beta_n * (1.0 - u) * u;  // Ruyten correction
    }
    else if (index == 2) {
        s = (1.0/6.0) * (3.0*pow(1.0 - u, 3) - 6.0*pow(1.0 - u, 2) + 4.0);
        s -= beta_n * (1.0 - u) * u;  // Ruyten correction
    }
    else if (index == 3) {
        s = (1.0/6.0) * (u*u*u);
    }

    // Check if the cell is below the axis
    if (index + ir < 0) {
        s *= flip;
    }
    return s;
}

'''

kernel_src = r'''
extern "C" __global__
void deposit_rho_gpu_linear_one_mode(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ w,
    const double q,
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    complex<double>* __restrict__ rho_m,
    const int m,
    const int* __restrict__ cell_idx,
    const int* __restrict__ prefix_sum,
    const double* __restrict__ beta_n,
    int Ncells)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Ncells) return;

    int iz_upper = i / (Nr+1);
    int ir_upper = i - iz_upper * (Nr+1);

    int incl_offset = prefix_sum[i];
    int frequency_per_cell = (i > 0) ? (incl_offset - prefix_sum[i-1]) : incl_offset;

    complex<double> R_m_00(0.0,0.0);
    complex<double> R_m_01(0.0,0.0);
    complex<double> R_m_10(0.0,0.0);
    complex<double> R_m_11(0.0,0.0);

    int f = (m % 2 == 0 ? 1 : -1);

    for (int j=0; j < frequency_per_cell; j++) {
        int ptcl_idx = incl_offset - 1 - j;

        double xj = x[ptcl_idx];
        double yj = y[ptcl_idx];
        double zj = z[ptcl_idx];
        double wj = q * w[ptcl_idx];

        double rj = sqrt(xj*xj + yj*yj);
        double cosv, sinv;
        if (rj != 0.0) {
            double invr = 1.0 / rj;
            cosv = xj * invr;
            sinv = yj * invr;
        } else {
            cosv = 1.0;
            sinv = 0.0;
        }

        complex<double> exptheta_m(1.0,0.0);
        complex<double> base(cosv, sinv);
        for (int k=0; k<m; k++) {
            exptheta_m *= base;
        }

        double r_cell = invdr * (rj - rmin) - 0.5;
        double z_cell = invdz * (zj - zmin) - 0.5;

        int ir = min(int(ceil(r_cell)), Nr);
        double bn = beta_n[ir];

        complex<double> R_m_scal = wj * exptheta_m;

        R_m_00 += Sr_linear(r_cell, 0, f, bn) * Sz_linear(z_cell, 0) * R_m_scal;
        R_m_01 += Sr_linear(r_cell, 0, f, bn) * Sz_linear(z_cell, 1) * R_m_scal;
        R_m_10 += Sr_linear(r_cell, 1, f, bn) * Sz_linear(z_cell, 0) * R_m_scal;
        R_m_11 += Sr_linear(r_cell, 1, f, bn) * Sz_linear(z_cell, 1) * R_m_scal;
    }

    int iz0 = iz_upper - 1;
    int iz1 = iz_upper;
    if (iz0 < 0) iz0 += Nz;

    int ir0 = ir_upper - 1;
    int ir1 = min(ir_upper, Nr-1);
    if (ir0 < 0) {
        ir0 = -(1 + ir0);
    }

    if (frequency_per_cell > 0) {

        double* rho_ptr = reinterpret_cast<double*>(rho_m);
        atomicAdd(&rho_ptr[2*(iz0 * Nr + ir0)],     R_m_00.real());
        atomicAdd(&rho_ptr[2*(iz0 * Nr + ir0) + 1], R_m_00.imag());

        atomicAdd(&rho_ptr[2*(iz0 * Nr + ir1)],     R_m_10.real());
        atomicAdd(&rho_ptr[2*(iz0 * Nr + ir1) + 1], R_m_10.imag());

        atomicAdd(&rho_ptr[2*(iz1 * Nr + ir0)],     R_m_01.real());
        atomicAdd(&rho_ptr[2*(iz1 * Nr + ir0) + 1], R_m_01.imag());

        atomicAdd(&rho_ptr[2*(iz1 * Nr + ir1)],     R_m_11.real());
        atomicAdd(&rho_ptr[2*(iz1 * Nr + ir1) + 1], R_m_11.imag());

}
}
extern "C" __global__
void deposit_J_gpu_linear_one_mode(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ w,
    const double q,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ inv_gamma,
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    complex<double>* __restrict__ j_r_m,
    complex<double>* __restrict__ j_t_m,
    complex<double>* __restrict__ j_z_m,
    const int m,
    const int* __restrict__ cell_idx,
    const int* __restrict__ prefix_sum,
    const double* __restrict__ beta_n,
    const double c,
    const int Ncells
){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= Ncells) return;

    // z,r index
    int iz_upper = i / (Nr+1);
    int ir_upper = i - iz_upper*(Nr+1);

    int incl_offset = prefix_sum[i];
    int frequency_per_cell = (i>0) ? (incl_offset - prefix_sum[i-1]) : incl_offset;

    // Accumulators
    complex<double> J_r_m_00(0,0), J_t_m_00(0,0), J_z_m_00(0,0);
    complex<double> J_r_m_01(0,0), J_t_m_01(0,0), J_z_m_01(0,0);
    complex<double> J_r_m_10(0,0), J_t_m_10(0,0), J_z_m_10(0,0);
    complex<double> J_r_m_11(0,0), J_t_m_11(0,0), J_z_m_11(0,0);

    int f = (m%2==0) ? 1 : -1;

    for (int j=0;j<frequency_per_cell;j++){
        int ptcl_idx = incl_offset-1-j;
        double xj = x[ptcl_idx];
        double yj = y[ptcl_idx];
        double zj = z[ptcl_idx];
        double uxj = ux[ptcl_idx];
        double uyj = uy[ptcl_idx];
        double uzj = uz[ptcl_idx];
        double inv_g = inv_gamma[ptcl_idx];
        double wj = q * w[ptcl_idx];

        double rj = sqrt(xj*xj + yj*yj);
        double cosv, sinv;
        if (rj!=0.0){
            cosv = xj/rj;
            sinv = yj/rj;
        }else{
            cosv = 1.0; sinv = 0.0;
        }

        // exp(i*m*theta)
        complex<double> exptheta_m(1.0,0.0);
        complex<double> tmp(cosv, sinv);
        for(int k=0;k<m;k++){
            exptheta_m *= tmp;
        }

        double r_cell = invdr*(rj - rmin) - 0.5;
        double z_cell = invdz*(zj - zmin) - 0.5;
        int ir = (int)ceil(r_cell);
        if (ir>Nr) ir=Nr;
        double bn = beta_n[ir];

        complex<double> J_r_m_scal = wj * c * inv_g*(cosv*uxj + sinv*uyj) * exptheta_m;
        complex<double> J_t_m_scal = wj * c * inv_g*(cosv*uyj - sinv*uxj) * exptheta_m;
        complex<double> J_z_m_scal = wj * c * inv_g*uzj * exptheta_m;

        J_r_m_00 += Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 0) * J_r_m_scal;
        J_t_m_00 += Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 0) * J_t_m_scal;
        J_z_m_00 += Sr_linear(r_cell, 0, f, bn)*Sz_linear(z_cell, 0) * J_z_m_scal;
        J_r_m_01 += Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 1) * J_r_m_scal;
        J_t_m_01 += Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 1) * J_t_m_scal;
        J_z_m_01 += Sr_linear(r_cell, 0, f, bn)*Sz_linear(z_cell, 1) * J_z_m_scal;

        J_r_m_10 += Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 0) * J_r_m_scal;
        J_t_m_10 += Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 0) * J_t_m_scal;
        J_z_m_10 += Sr_linear(r_cell, 1, f, bn)*Sz_linear(z_cell, 0) * J_z_m_scal;
        J_r_m_11 += Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 1) * J_r_m_scal;
        J_t_m_11 += Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 1) * J_t_m_scal;
        J_z_m_11 += Sr_linear(r_cell, 1, f, bn)*Sz_linear(z_cell, 1) * J_z_m_scal;
    }

    int iz0 = iz_upper - 1;
    int iz1 = iz_upper;
    if (iz0<0) iz0+=Nz;
    int ir0 = ir_upper - 1;
    int ir1 = (ir_upper<Nr)? ir_upper : (Nr-1);
    if (ir0<0) ir0 = -(1+ir0);

    int stride = Nr;
    // ---- atomicAdd to j_r_m/j_t_m/j_z_m ----
    // real part always
    double* j_r_m_x = reinterpret_cast<double*>(j_r_m);
    double* j_t_m_x = reinterpret_cast<double*>(j_t_m);
    double* j_z_m_x = reinterpret_cast<double*>(j_z_m);
    atomicAdd(&j_r_m_x[2*(iz0*stride+ir0)], J_r_m_00.real());
    atomicAdd(&j_r_m_x[2*(iz0*stride+ir1)], J_r_m_10.real());
    atomicAdd(&j_r_m_x[2*(iz1*stride+ir0)], J_r_m_01.real());
    atomicAdd(&j_r_m_x[2*(iz1*stride+ir1)], J_r_m_11.real());

    atomicAdd(&j_t_m_x[2*(iz0*stride+ir0)], J_t_m_00.real());
    atomicAdd(&j_t_m_x[2*(iz0*stride+ir1)], J_t_m_10.real());
    atomicAdd(&j_t_m_x[2*(iz1*stride+ir0)], J_t_m_01.real());
    atomicAdd(&j_t_m_x[2*(iz1*stride+ir1)], J_t_m_11.real());

    atomicAdd(&j_z_m_x[2*(iz0*stride+ir0)], J_z_m_00.real());
    atomicAdd(&j_z_m_x[2*(iz0*stride+ir1)], J_z_m_10.real());
    atomicAdd(&j_z_m_x[2*(iz1*stride+ir0)], J_z_m_01.real());
    atomicAdd(&j_z_m_x[2*(iz1*stride+ir1)], J_z_m_11.real());

    if (m>0){
        atomicAdd(&j_r_m_x[2*(iz0*stride+ir0)+1], J_r_m_00.imag());
        atomicAdd(&j_r_m_x[2*(iz0*stride+ir1)+1], J_r_m_10.imag());
        atomicAdd(&j_r_m_x[2*(iz1*stride+ir0)+1], J_r_m_01.imag());
        atomicAdd(&j_r_m_x[2*(iz1*stride+ir1)+1], J_r_m_11.imag());

        atomicAdd(&j_t_m_x[2*(iz0*stride+ir0)+1], J_t_m_00.imag());
        atomicAdd(&j_t_m_x[2*(iz0*stride+ir1)+1], J_t_m_10.imag());
        atomicAdd(&j_t_m_x[2*(iz1*stride+ir0)+1], J_t_m_01.imag());
        atomicAdd(&j_t_m_x[2*(iz1*stride+ir1)+1], J_t_m_11.imag());

        atomicAdd(&j_z_m_x[2*(iz0*stride+ir0)+1], J_z_m_00.imag());
        atomicAdd(&j_z_m_x[2*(iz0*stride+ir1)+1], J_z_m_10.imag());
        atomicAdd(&j_z_m_x[2*(iz1*stride+ir0)+1], J_z_m_01.imag());
        atomicAdd(&j_z_m_x[2*(iz1*stride+ir1)+1], J_z_m_11.imag());
    }
}

extern "C" __global__
void deposit_rho_gpu_cubic_one_mode(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ w,
    const double q,
    const double invdz,
    const double zmin,
    const int    Nz,
    const double invdr,
    const double rmin,
    const int    Nr,
    complex<double>* __restrict__ rho_m,
    const int m,
    const int* __restrict__ cell_idx,     // (not used, kept for signature parity)
    const int* __restrict__ prefix_sum,
    const double* __restrict__ beta_n,
    const int Ncells)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Ncells) return;

    // Map prefix index -> upper grid points (see original numba code)
    int iz_upper = i / (Nr + 1);
    int ir_upper = i - iz_upper * (Nr + 1);

    int incl_offset = prefix_sum[i];
    int frequency_per_cell = (i > 0) ? (incl_offset - prefix_sum[i-1]) : incl_offset;

    // Accumulators (16 surrounding cells)
    complex<double> R_m_00(0.0,0.0), R_m_01(0.0,0.0), R_m_02(0.0,0.0), R_m_03(0.0,0.0);
    complex<double> R_m_10(0.0,0.0), R_m_11(0.0,0.0), R_m_12(0.0,0.0), R_m_13(0.0,0.0);
    complex<double> R_m_20(0.0,0.0), R_m_21(0.0,0.0), R_m_22(0.0,0.0), R_m_23(0.0,0.0);
    complex<double> R_m_30(0.0,0.0), R_m_31(0.0,0.0), R_m_32(0.0,0.0), R_m_33(0.0,0.0);

    // Sign for the shape factors f = (-1)^m
    int f = (m % 2 == 0) ? 1 : -1;

    for (int j = 0; j < frequency_per_cell; ++j) {
        int ptcl_idx = incl_offset - 1 - j; // inclusive cumulative sum

        // Positions & weight
        double xj = x[ptcl_idx];
        double yj = y[ptcl_idx];
        double zj = z[ptcl_idx];
        double wj = q * w[ptcl_idx];

        // Cylindrical conversion
        double rj = sqrt(xj*xj + yj*yj);
        double cos_th, sin_th;
        if (rj != 0.0) {
            double invr = 1.0 / rj;
            cos_th = xj * invr;
            sin_th = yj * invr;
        } else {
            cos_th = 1.0;
            sin_th = 0.0;
        }

        // Azimuthal factor (cos + i sin)^m
        complex<double> exptheta_m(1.0, 0.0);
        complex<double> base(cos_th, sin_th);
        for (int k = 0; k < m; ++k) {
            exptheta_m *= base;
        }

        // Particle position in cell units
        double r_cell = invdr * (rj - rmin) - 0.5;
        double z_cell = invdz * (zj - zmin) - 0.5;

        // Ruyten-corrected coefficient beta_n[ir]
        int ir_b = (int)ceil(r_cell);
        if (ir_b > Nr) ir_b = Nr;
        double bn = beta_n[ir_b];

        // Charge contribution
        complex<double> R_m_scal = exptheta_m * wj;

        // 16 weights (r index 0..3, z index 0..3)
        // r: Sr_cubic(r_cell, idx_r, f, bn)
        // z: Sz_cubic(z_cell, idx_z)
        // Accumulate: R_m_rz += Sr*Sz * R_m_scal
        double Sr0 = Sr_cubic(r_cell, 0, f, bn);
        double Sr1 = Sr_cubic(r_cell, 1, f, bn);
        double Sr2 = Sr_cubic(r_cell, 2, f, bn);
        double Sr3 = Sr_cubic(r_cell, 3, f, bn);

        double Sz0 = Sz_cubic(z_cell, 0);
        double Sz1 = Sz_cubic(z_cell, 1);
        double Sz2 = Sz_cubic(z_cell, 2);
        double Sz3 = Sz_cubic(z_cell, 3);

        R_m_00 += (Sr0 * Sz0) * R_m_scal;
        R_m_01 += (Sr0 * Sz1) * R_m_scal;
        R_m_02 += (Sr0 * Sz2) * R_m_scal;
        R_m_03 += (Sr0 * Sz3) * R_m_scal;

        R_m_10 += (Sr1 * Sz0) * R_m_scal;
        R_m_11 += (Sr1 * Sz1) * R_m_scal;
        R_m_12 += (Sr1 * Sz2) * R_m_scal;
        R_m_13 += (Sr1 * Sz3) * R_m_scal;

        R_m_20 += (Sr2 * Sz0) * R_m_scal;
        R_m_21 += (Sr2 * Sz1) * R_m_scal;
        R_m_22 += (Sr2 * Sz2) * R_m_scal;
        R_m_23 += (Sr2 * Sz3) * R_m_scal;

        R_m_30 += (Sr3 * Sz0) * R_m_scal;
        R_m_31 += (Sr3 * Sz1) * R_m_scal;
        R_m_32 += (Sr3 * Sz2) * R_m_scal;
        R_m_33 += (Sr3 * Sz3) * R_m_scal;
    }

    // Indices for deposition
    int iz0 = iz_upper - 2;
    int iz1 = iz_upper - 1;
    int iz2 = iz_upper;
    int iz3 = iz_upper + 1;
    if (iz0 < 0) iz0 += Nz;
    if (iz1 < 0) iz1 += Nz;
    if (iz3 > Nz - 1) iz3 -= Nz;

    int ir0 = ir_upper - 2;
    int ir1 = min(ir_upper - 1, Nr - 1);
    int ir2 = min(ir_upper    , Nr - 1);
    int ir3 = min(ir_upper + 1, Nr - 1);
    if (ir0 < 0) ir0 = -(1 + ir0);
    if (ir1 < 0) ir1 = -(1 + ir1);

    if (frequency_per_cell > 0) {
        // Atomic add on real part always; imag only when m>0
        double* __restrict__ rho_ptr = reinterpret_cast<double*>(rho_m);
        // helper lambda avoided per user's instruction; write explicitly
        // Each complex is [real, imag] in memory
        int idx00 = iz0 * Nr + ir0; int idx10 = iz0 * Nr + ir1; int idx20 = iz0 * Nr + ir2; int idx30 = iz0 * Nr + ir3;
        int idx01 = iz1 * Nr + ir0; int idx11 = iz1 * Nr + ir1; int idx21 = iz1 * Nr + ir2; int idx31 = iz1 * Nr + ir3;
        int idx02 = iz2 * Nr + ir0; int idx12 = iz2 * Nr + ir1; int idx22 = iz2 * Nr + ir2; int idx32 = iz2 * Nr + ir3;
        int idx03 = iz3 * Nr + ir0; int idx13 = iz3 * Nr + ir1; int idx23 = iz3 * Nr + ir2; int idx33 = iz3 * Nr + ir3;

        // real parts
        atomicAdd(&rho_ptr[2*idx00], R_m_00.real());
        atomicAdd(&rho_ptr[2*idx10], R_m_10.real());
        atomicAdd(&rho_ptr[2*idx20], R_m_20.real());
        atomicAdd(&rho_ptr[2*idx30], R_m_30.real());
        atomicAdd(&rho_ptr[2*idx01], R_m_01.real());
        atomicAdd(&rho_ptr[2*idx11], R_m_11.real());
        atomicAdd(&rho_ptr[2*idx21], R_m_21.real());
        atomicAdd(&rho_ptr[2*idx31], R_m_31.real());
        atomicAdd(&rho_ptr[2*idx02], R_m_02.real());
        atomicAdd(&rho_ptr[2*idx12], R_m_12.real());
        atomicAdd(&rho_ptr[2*idx22], R_m_22.real());
        atomicAdd(&rho_ptr[2*idx32], R_m_32.real());
        atomicAdd(&rho_ptr[2*idx03], R_m_03.real());
        atomicAdd(&rho_ptr[2*idx13], R_m_13.real());
        atomicAdd(&rho_ptr[2*idx23], R_m_23.real());
        atomicAdd(&rho_ptr[2*idx33], R_m_33.real());

        if (m > 0) {
            atomicAdd(&rho_ptr[2*idx00 + 1], R_m_00.imag());
            atomicAdd(&rho_ptr[2*idx10 + 1], R_m_10.imag());
            atomicAdd(&rho_ptr[2*idx20 + 1], R_m_20.imag());
            atomicAdd(&rho_ptr[2*idx30 + 1], R_m_30.imag());
            atomicAdd(&rho_ptr[2*idx01 + 1], R_m_01.imag());
            atomicAdd(&rho_ptr[2*idx11 + 1], R_m_11.imag());
            atomicAdd(&rho_ptr[2*idx21 + 1], R_m_21.imag());
            atomicAdd(&rho_ptr[2*idx31 + 1], R_m_31.imag());
            atomicAdd(&rho_ptr[2*idx02 + 1], R_m_02.imag());
            atomicAdd(&rho_ptr[2*idx12 + 1], R_m_12.imag());
            atomicAdd(&rho_ptr[2*idx22 + 1], R_m_22.imag());
            atomicAdd(&rho_ptr[2*idx32 + 1], R_m_32.imag());
            atomicAdd(&rho_ptr[2*idx03 + 1], R_m_03.imag());
            atomicAdd(&rho_ptr[2*idx13 + 1], R_m_13.imag());
            atomicAdd(&rho_ptr[2*idx23 + 1], R_m_23.imag());
            atomicAdd(&rho_ptr[2*idx33 + 1], R_m_33.imag());
        }
    }
}




extern "C"  __global__ void deposit_J_gpu_cubic_one_mode(
    // particle data
    const double* x, const double* y, const double* z,
    const double* w, const double  q,
    const double* ux, const double* uy, const double* uz,
    const double* inv_gamma,
    // grid
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    // output fields (complex<double> arrays, shape (Nz, Nr) flattened row-major)
    complex<double>* j_r_m,
    complex<double>* j_t_m,
    complex<double>* j_z_m,
    // mode and helpers
    const int m,
    const int* cell_idx,          // kept for interface parity (unused in body)
    const int* prefix_sum,        // inclusive prefix sum, length = n_cells
    const double* beta_n,
    const double c,
    const int Ncells   
) {
    // 1D grid
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Ncells) {

        // iz_upper and ir_upper
        int iz_upper = i / (Nr + 1);
        int ir_upper = i - iz_upper * (Nr + 1);

        // inclusive offset and frequency_per_cell
        int incl_offset = prefix_sum[i];
        int frequency_per_cell;
        if (i > 0) {
            frequency_per_cell = incl_offset - prefix_sum[i - 1];
        } 
        else {
            frequency_per_cell = incl_offset;
        }

        // local complex accumulators (16 per component)
        complex<double> J_r_m_00 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_00 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_00 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_01 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_01 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_01 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_02 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_02 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_02 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_03 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_03 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_03 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_10 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_10 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_10 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_11 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_11 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_11 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_12 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_12 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_12 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_13 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_13 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_13 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_20 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_20 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_20 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_21 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_21 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_21 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_22 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_22 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_22 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_23 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_23 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_23 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_30 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_30 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_30 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_31 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_31 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_31 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_32 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_32 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_32 = complex<double>(0.0, 0.0);

        complex<double> J_r_m_33 = complex<double>(0.0, 0.0);
        complex<double> J_t_m_33 = complex<double>(0.0, 0.0);
        complex<double> J_z_m_33 = complex<double>(0.0, 0.0);

        // f = (-1)**m
        int f = (m%2==0) ? 1 : -1;

        // loop over particles in cell
        for (int j = 0; j < frequency_per_cell; ++j) {
            int ptcl_idx = incl_offset - 1 - j;

            double xj = x[ptcl_idx];
            double yj = y[ptcl_idx];
            double zj = z[ptcl_idx];

            double uxj = ux[ptcl_idx];
            double uyj = uy[ptcl_idx];
            double uzj = uz[ptcl_idx];

            double inv_gammaj = inv_gamma[ptcl_idx];
            double wj = q * w[ptcl_idx];

            double rj = sqrt(xj*xj + yj*yj);
            double cosv = 1.0, sinv = 0.0;
            if (rj != 0.0) {
                double invr = 1.0 / rj;
                cosv = xj * invr;
                sinv = yj * invr;
            }

            // exptheta_m = (cos + i sin)^m via repeated multiply (same as original)
            complex<double> exptheta_m(1.0,0.0);
            complex<double> base(cosv, sinv);
            for (int k=0; k<m; k++) {
                exptheta_m *= base;
            }

            double r_cell = invdr * (rj - rmin) - 0.5;
            double z_cell = invdz * (zj - zmin) - 0.5;

            int ir = ceil(r_cell);
            if (ir > Nr) ir = Nr;
            double bn = beta_n[ir];

            // J scalars times complex azimuth factor
            complex<double> J_r_m_scal = wj * c * inv_gammaj * (cosv*uxj + sinv*uyj) * exptheta_m;
            complex<double> J_t_m_scal = wj * c * inv_gammaj * (cosv*uyj - sinv*uxj) * exptheta_m;
            complex<double> J_z_m_scal = wj * c * inv_gammaj * uzj * exptheta_m;

            // accumulate (exact same expressions as Numba version)
            J_r_m_00 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 0) * J_r_m_scal;
            J_r_m_01 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 1) * J_r_m_scal;
            J_r_m_02 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 2) * J_r_m_scal;
            J_r_m_03 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 3) * J_r_m_scal;

            J_r_m_10 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 0) * J_r_m_scal;
            J_r_m_11 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 1) * J_r_m_scal;
            J_r_m_12 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 2) * J_r_m_scal;
            J_r_m_13 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 3) * J_r_m_scal;

            J_r_m_20 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 0) * J_r_m_scal;
            J_r_m_21 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 1) * J_r_m_scal;
            J_r_m_22 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 2) * J_r_m_scal;
            J_r_m_23 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 3) * J_r_m_scal;

            J_r_m_30 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 0) * J_r_m_scal;
            J_r_m_31 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 1) * J_r_m_scal;
            J_r_m_32 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 2) * J_r_m_scal;
            J_r_m_33 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 3) * J_r_m_scal;

            J_t_m_00 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 0) * J_t_m_scal;
            J_t_m_01 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 1) * J_t_m_scal;
            J_t_m_02 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 2) * J_t_m_scal;
            J_t_m_03 += Sr_cubic(r_cell, 0, -f, bn) * Sz_cubic(z_cell, 3) * J_t_m_scal;

            J_t_m_10 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 0) * J_t_m_scal;
            J_t_m_11 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 1) * J_t_m_scal;
            J_t_m_12 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 2) * J_t_m_scal;
            J_t_m_13 += Sr_cubic(r_cell, 1, -f, bn) * Sz_cubic(z_cell, 3) * J_t_m_scal;

            J_t_m_20 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 0) * J_t_m_scal;
            J_t_m_21 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 1) * J_t_m_scal;
            J_t_m_22 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 2) * J_t_m_scal;
            J_t_m_23 += Sr_cubic(r_cell, 2, -f, bn) * Sz_cubic(z_cell, 3) * J_t_m_scal;

            J_t_m_30 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 0) * J_t_m_scal;
            J_t_m_31 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 1) * J_t_m_scal;
            J_t_m_32 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 2) * J_t_m_scal;
            J_t_m_33 += Sr_cubic(r_cell, 3, -f, bn) * Sz_cubic(z_cell, 3) * J_t_m_scal;

            J_z_m_00 += Sr_cubic(r_cell, 0, f, bn) * Sz_cubic(z_cell, 0) * J_z_m_scal;
            J_z_m_01 += Sr_cubic(r_cell, 0, f, bn) * Sz_cubic(z_cell, 1) * J_z_m_scal;
            J_z_m_02 += Sr_cubic(r_cell, 0, f, bn) * Sz_cubic(z_cell, 2) * J_z_m_scal;
            J_z_m_03 += Sr_cubic(r_cell, 0, f, bn) * Sz_cubic(z_cell, 3) * J_z_m_scal;

            J_z_m_10 += Sr_cubic(r_cell, 1, f, bn) * Sz_cubic(z_cell, 0) * J_z_m_scal;
            J_z_m_11 += Sr_cubic(r_cell, 1, f, bn) * Sz_cubic(z_cell, 1) * J_z_m_scal;
            J_z_m_12 += Sr_cubic(r_cell, 1, f, bn) * Sz_cubic(z_cell, 2) * J_z_m_scal;
            J_z_m_13 += Sr_cubic(r_cell, 1, f, bn) * Sz_cubic(z_cell, 3) * J_z_m_scal;

            J_z_m_20 += Sr_cubic(r_cell, 2, f, bn) * Sz_cubic(z_cell, 0) * J_z_m_scal;
            J_z_m_21 += Sr_cubic(r_cell, 2, f, bn) * Sz_cubic(z_cell, 1) * J_z_m_scal;
            J_z_m_22 += Sr_cubic(r_cell, 2, f, bn) * Sz_cubic(z_cell, 2) * J_z_m_scal;
            J_z_m_23 += Sr_cubic(r_cell, 2, f, bn) * Sz_cubic(z_cell, 3) * J_z_m_scal;

            J_z_m_30 += Sr_cubic(r_cell, 3, f, bn) * Sz_cubic(z_cell, 0) * J_z_m_scal;
            J_z_m_31 += Sr_cubic(r_cell, 3, f, bn) * Sz_cubic(z_cell, 1) * J_z_m_scal;
            J_z_m_32 += Sr_cubic(r_cell, 3, f, bn) * Sz_cubic(z_cell, 2) * J_z_m_scal;
            J_z_m_33 += Sr_cubic(r_cell, 3, f, bn) * Sz_cubic(z_cell, 3) * J_z_m_scal;
        } // end per-particle loop

        // compute iz/ir target indices (same as python)
        int iz0 = iz_upper - 2;
        int iz1 = iz_upper - 1;
        int iz2 = iz_upper;
        int iz3 = iz_upper + 1;
        if (iz0 < 0) iz0 += Nz;
        if (iz1 < 0) iz1 += Nz;
        if (iz3 > Nz - 1) iz3 -= Nz;

        int ir0 = ir_upper - 2;
        int ir1 = (ir_upper - 1 < Nr) ? (ir_upper - 1) : (Nr - 1);
        int ir2 = (ir_upper < Nr) ? ir_upper : (Nr - 1);
        int ir3 = (ir_upper + 1 < Nr) ? (ir_upper + 1) : (Nr - 1);
        if (ir0 < 0) ir0 = -(1 + ir0);
        if (ir1 < 0) ir1 = -(1 + ir1);

        // atomic add to global memory (complex arrays) — keep imag only if m>0
        if (frequency_per_cell > 0) {
            int izs[4] = {iz0, iz1, iz2, iz3};
            int irs[4] = {ir0, ir1, ir2, ir3};

            // map local accumulators into arrays for iteration
            complex<double> Jr[4][4] = {
                {J_r_m_00, J_r_m_01, J_r_m_02, J_r_m_03},
                {J_r_m_10, J_r_m_11, J_r_m_12, J_r_m_13},
                {J_r_m_20, J_r_m_21, J_r_m_22, J_r_m_23},
                {J_r_m_30, J_r_m_31, J_r_m_32, J_r_m_33}
            };
            complex<double> Jt[4][4] = {
                {J_t_m_00, J_t_m_01, J_t_m_02, J_t_m_03},
                {J_t_m_10, J_t_m_11, J_t_m_12, J_t_m_13},
                {J_t_m_20, J_t_m_21, J_t_m_22, J_t_m_23},
                {J_t_m_30, J_t_m_31, J_t_m_32, J_t_m_33}
            };
            complex<double> Jz[4][4] = {
                {J_z_m_00, J_z_m_01, J_z_m_02, J_z_m_03},
                {J_z_m_10, J_z_m_11, J_z_m_12, J_z_m_13},
                {J_z_m_20, J_z_m_21, J_z_m_22, J_z_m_23},
                {J_z_m_30, J_z_m_31, J_z_m_32, J_z_m_33}
            };

            for (int zz = 0; zz < 4; ++zz) {
                int iz = izs[zz];
                for (int rr = 0; rr < 4; ++rr) {
                    int ir = irs[rr];
                    int idx = iz * Nr + ir;

                    // j_r
                    double* pr = reinterpret_cast<double*>(&j_r_m[idx]);
                    atomicAdd(&pr[0], real(Jr[rr][zz]) );
                    if (m > 0) atomicAdd(&pr[1], imag(Jr[rr][zz]) );

                    // j_t
                    double* pt = reinterpret_cast<double*>(&j_t_m[idx]);
                    atomicAdd(&pt[0], real(Jt[rr][zz]) );
                    if (m > 0) atomicAdd(&pt[1], imag(Jt[rr][zz]) );

                    // j_z
                    double* pz = reinterpret_cast<double*>(&j_z_m[idx]);
                    atomicAdd(&pz[0], real(Jz[rr][zz]) );
                    if (m > 0) atomicAdd(&pz[1], imag(Jz[rr][zz]) );
                }
            }
        } // end if freq>0
    } // end if i < prefix_sum_len
} // end kernel
'''



full_src = aux_src + kernel_src
deposit_rho_gpu_linear_one_mode_c = cp.RawKernel(full_src, "deposit_rho_gpu_linear_one_mode")
deposit_J_gpu_linear_one_mode_c = cp.RawKernel(full_src, "deposit_J_gpu_linear_one_mode")
deposit_rho_gpu_cubic_one_mode_c = cp.RawKernel(full_src, "deposit_rho_gpu_cubic_one_mode")
deposit_J_gpu_cubic_one_mode_c = cp.RawKernel(full_src, "deposit_J_gpu_cubic_one_mode")
