# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
"""


# JIT-compilation of particle shapes

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
'''
kernel_src = r'''

extern "C" __global__ void deposit_rho_gpu_unsorted(
    // 粒子
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ w,
    const int q,
    // 网格与几何
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    // 输出：rho_m (Nz, Nr) 的复数网格（展平后按行主序）
    complex<double>* __restrict__ rho_m,
    const int m,
    double beta_n, // 长度 >= Nr+1
    const int    n_particles
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_particles) return;

    // 形函数符号：f = (-1)^m
    int f = ((m & 1) != 0) ? -1 : 1;

    // 位置与权重
    double xj = x[i];
    double yj = y[i];
    double zj = z[i];
    double wj = q * w[i];

    // 极坐标与方位因子
    double rj = sqrt(xj*xj + yj*yj);
    double cosv = 1.0, sinv = 0.0;
    if (rj != 0.0){
        double invr = 1.0 / rj;
        cosv = xj * invr;
        sinv = yj * invr;
    }
    // exptheta_m = (cos + i sin)^m，按原代码用循环相乘
    complex<double> exptheta_m = complex<double>(1.0, 0.0);
    for (int mm = 0; mm < m; ++mm){
        exptheta_m *= complex<double>(cosv, sinv);
    }

    // 单元坐标
    double r_cell = invdr * (rj - rmin) - 0.5;
    double z_cell = invdz * (zj - zmin) - 0.5;

    // 上界单元索引
    int ir_up = (int)ceil(r_cell);
    if (ir_up > Nr) ir_up = Nr;          // 与原 numba: min(ceil, Nr)
    if (ir_up < 0)  ir_up = 0;
    int iz_up = (int)ceil(z_cell);

    // 周期边界（z）
    if (iz_up < 0)      iz_up += Nz;
    else if (iz_up >= Nz) iz_up -= Nz;

    // 需要累加的四个目标网格点 (z 两点 × r 两点)
    int iz0 = iz_up - 1;
    int iz1 = iz_up;
    if (iz0 < 0) iz0 += Nz;

    int ir0 = ir_up - 1;
    int ir1 = (ir_up < Nr) ? ir_up : (Nr - 1);
    if (ir0 < 0){
        // 轴下折叠
        ir0 = -(1 + ir0);
    }

    // Ruyten 系数（按原实现：取 ir_up 处）
    int ir_beta = ir_up;
    if (ir_beta < 0) ir_beta = 0;
    if (ir_beta > Nr) ir_beta = Nr;

    // ρ 的标量系数（不含形函数）
    complex<double> R_m_scal = complex<double>(wj, 0.0) * exptheta_m;

    // 线性形函数（完全展开，遵循你的“不要循环优化”的要求）
    double Sr00 = Sr_linear(r_cell, 0, f, beta_n);
    double Sr10 = Sr_linear(r_cell, 1, f, beta_n);
    double Sz00 = Sz_linear(z_cell, 0);
    double Sz01 = Sz_linear(z_cell, 1);

    complex<double> R_m_00 = complex<double>(Sr00 * Sz00, 0.0) * R_m_scal;
    complex<double> R_m_01 = complex<double>(Sr00 * Sz01, 0.0) * R_m_scal;
    complex<double> R_m_10 = complex<double>(Sr10 * Sz00, 0.0) * R_m_scal;
    complex<double> R_m_11 = complex<double>(Sr10 * Sz01, 0.0) * R_m_scal;

    // 扁平化索引
    int idx00 = iz0 * Nr + ir0;
    int idx10 = iz0 * Nr + ir1;
    int idx01 = iz1 * Nr + ir0;
    int idx11 = iz1 * Nr + ir1;

    // 原子加到全局复数数组（只在 m>0 时累加虚部）
    // (将 complex<double>* reinterpret_cast 为 double*，依次累加 [real, imag])
    // -- (iz0, ir0)
    {
        double* p = reinterpret_cast<double*>(&rho_m[idx00]);
        atomicAdd(&p[0], real(R_m_00));
        if (m > 0) atomicAdd(&p[1], imag(R_m_00));
    }
    // -- (iz0, ir1)
    {
        double* p = reinterpret_cast<double*>(&rho_m[idx10]);
        atomicAdd(&p[0], real(R_m_10));
        if (m > 0) atomicAdd(&p[1], imag(R_m_10));
    }
    // -- (iz1, ir0)
    {
        double* p = reinterpret_cast<double*>(&rho_m[idx01]);
        atomicAdd(&p[0], real(R_m_01));
        if (m > 0) atomicAdd(&p[1], imag(R_m_01));
    }
    // -- (iz1, ir1)
    {
        double* p = reinterpret_cast<double*>(&rho_m[idx11]);
        atomicAdd(&p[0], real(R_m_11));
        if (m > 0) atomicAdd(&p[1], imag(R_m_11));
    }
}

extern "C" __global__
void deposit_J_gpu_unsorted(
    const double* x,
    const double* y,
    const double* z,
    const double* w,
    const int  q,
    const double* vx,
    const double* vy,
    const double* vz,
    const double  invdz, const double zmin, const int Nz,
    const double  invdr, const double rmin, const int Nr,
    complex<double>* j_r_m,
    complex<double>* j_t_m,
    complex<double>* j_z_m,
    const int m,
    double* beta_n,
    const int Np // number of particles (size of w)
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Np) return;

    // Sign for shape factors
    double f = ( (m & 1) == 0 ) ? 1.0 : -1.0;

    // Particle data
    double xj = x[i];
    double yj = y[i];
    double zj = z[i];
    double vxj = vx[i];
    double vyj = vy[i];
    double vzj = vz[i];
    double wj = q * w[i];

    // Cylindrical conversion
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

    // azimuthal factor exptheta_m
    complex<double> exptheta_m(1.0, 0.0);
    for (int kk = 0; kk < m; ++kk) {
        exptheta_m *= complex<double>(cosv, sinv);
    }

    // positions in cell units
    double r_cell = invdr * (rj - rmin) - 0.5;
    double z_cell = invdz * (zj - zmin) - 0.5;

    // upper cell indices
    int ir = min((int)ceil(r_cell), Nr);
    int iz = (int)ceil(z_cell);
    // handle periodic boundaries in z
    if (iz < 0) iz += Nz;
    else if (iz >= Nz) iz -= Nz;

    int iz0 = iz - 1;
    int iz1 = iz;
    if (iz0 < 0) iz0 += Nz;

    int ir0 = ir - 1;
    int ir1 = min(ir, Nr - 1);
    if (ir0 < 0) ir0 = -(1 + ir0); // fold below axis

    double bn = beta_n[ir];

    // compute currents (complex)
    complex<double> J_r_m_scal = wj * (cosv*vxj + sinv*vyj) * exptheta_m;
    complex<double> J_t_m_scal = wj * (cosv*vyj - sinv*vxj) * exptheta_m;
    complex<double> J_z_m_scal = wj * vzj * exptheta_m;

    // contributions to 4 nearby nodes (linear)
    complex<double> J_r_m_00 = Sr_linear(r_cell, 0, -f, bn) * Sz_linear(z_cell, 0) * J_r_m_scal;
    complex<double> J_t_m_00 = Sr_linear(r_cell, 0, -f, bn) * Sz_linear(z_cell, 0) * J_t_m_scal;
    complex<double> J_z_m_00 = Sr_linear(r_cell, 0,  f, bn) * Sz_linear(z_cell, 0) * J_z_m_scal;

    complex<double> J_r_m_01 = Sr_linear(r_cell, 0, -f, bn) * Sz_linear(z_cell, 1) * J_r_m_scal;
    complex<double> J_t_m_01 = Sr_linear(r_cell, 0, -f, bn) * Sz_linear(z_cell, 1) * J_t_m_scal;
    complex<double> J_z_m_01 = Sr_linear(r_cell, 0,  f, bn) * Sz_linear(z_cell, 1) * J_z_m_scal;

    complex<double> J_r_m_10 = Sr_linear(r_cell, 1, -f, bn) * Sz_linear(z_cell, 0) * J_r_m_scal;
    complex<double> J_t_m_10 = Sr_linear(r_cell, 1, -f, bn) * Sz_linear(z_cell, 0) * J_t_m_scal;
    complex<double> J_z_m_10 = Sr_linear(r_cell, 1,  f, bn) * Sz_linear(z_cell, 0) * J_z_m_scal;

    complex<double> J_r_m_11 = Sr_linear(r_cell, 1, -f, bn) * Sz_linear(z_cell, 1) * J_r_m_scal;
    complex<double> J_t_m_11 = Sr_linear(r_cell, 1, -f, bn) * Sz_linear(z_cell, 1) * J_t_m_scal;
    complex<double> J_z_m_11 = Sr_linear(r_cell, 1,  f, bn) * Sz_linear(z_cell, 1) * J_z_m_scal;

    // linear indices
    int idx00 = iz0 * Nr + ir0;
    int idx10 = iz0 * Nr + ir1;
    int idx01 = iz1 * Nr + ir0;
    int idx11 = iz1 * Nr + ir1;

    // reinterpret complex arrays as double* (real, imag interleaved)
    double* j_r_m_x = reinterpret_cast<double*>(j_r_m);
    double* j_t_m_x = reinterpret_cast<double*>(j_t_m);
    double* j_z_m_x = reinterpret_cast<double*>(j_z_m);

    // atomic add real parts
    atomicAdd(&j_r_m_x[2 * idx00 + 0], J_r_m_00.real());
    atomicAdd(&j_r_m_x[2 * idx10 + 0], J_r_m_10.real());
    atomicAdd(&j_r_m_x[2 * idx01 + 0], J_r_m_01.real());
    atomicAdd(&j_r_m_x[2 * idx11 + 0], J_r_m_11.real());

    atomicAdd(&j_t_m_x[2 * idx00 + 0], J_t_m_00.real());
    atomicAdd(&j_t_m_x[2 * idx10 + 0], J_t_m_10.real());
    atomicAdd(&j_t_m_x[2 * idx01 + 0], J_t_m_01.real());
    atomicAdd(&j_t_m_x[2 * idx11 + 0], J_t_m_11.real());

    atomicAdd(&j_z_m_x[2 * idx00 + 0], J_z_m_00.real());
    atomicAdd(&j_z_m_x[2 * idx10 + 0], J_z_m_10.real());
    atomicAdd(&j_z_m_x[2 * idx01 + 0], J_z_m_01.real());
    atomicAdd(&j_z_m_x[2 * idx11 + 0], J_z_m_11.real());

    // atomic add imaginary parts only if m > 0
    if (m > 0) {
        atomicAdd(&j_r_m_x[2 * idx00 + 1], J_r_m_00.imag());
        atomicAdd(&j_r_m_x[2 * idx10 + 1], J_r_m_10.imag());
        atomicAdd(&j_r_m_x[2 * idx01 + 1], J_r_m_01.imag());
        atomicAdd(&j_r_m_x[2 * idx11 + 1], J_r_m_11.imag());

        atomicAdd(&j_t_m_x[2 * idx00 + 1], J_t_m_00.imag());
        atomicAdd(&j_t_m_x[2 * idx10 + 1], J_t_m_10.imag());
        atomicAdd(&j_t_m_x[2 * idx01 + 1], J_t_m_01.imag());
        atomicAdd(&j_t_m_x[2 * idx11 + 1], J_t_m_11.imag());

        atomicAdd(&j_z_m_x[2 * idx00 + 1], J_z_m_00.imag());
        atomicAdd(&j_z_m_x[2 * idx10 + 1], J_z_m_10.imag());
        atomicAdd(&j_z_m_x[2 * idx01 + 1], J_z_m_01.imag());
        atomicAdd(&j_z_m_x[2 * idx11 + 1], J_z_m_11.imag());
    }
}


'''
cuda_src = aux_src + kernel_src
# ---------------- 编译 RawModule 并取出 kernel ----------------
module = cp.RawModule(code=cuda_src)
deposit_rho_gpu_unsorted_c = module.get_function('deposit_rho_gpu_unsorted')
deposit_J_gpu_unsorted_c = module.get_function('deposit_J_gpu_unsorted')
