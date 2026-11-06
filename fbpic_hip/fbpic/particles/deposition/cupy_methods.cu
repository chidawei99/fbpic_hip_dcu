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


extern "C" __global__
void deposit_rho_gpu_linear(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ w,
    const double q,
    const double invdz, const double zmin, const int Nz,
    const double invdr, const double rmin, const int Nr,
    complex<double>* __restrict__ rho_m0,
    complex<double>* __restrict__ rho_m1,
    const int* __restrict__ cell_idx,
    const int* __restrict__ prefix_sum,
    const double* __restrict__ beta_n_m0,
    const double* __restrict__ beta_n_m1,
    const int Ncells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ncells) return;

    int iz_upper = i / (Nr + 1);
    int ir_upper = i % (Nr + 1);
    int incl_offset = prefix_sum[i];
    int freq = (i > 0) ? (incl_offset - prefix_sum[i - 1]) : incl_offset;

    complex<double> R_m0_00 = complex<double>(0.0, 0.0);
    complex<double> R_m0_01 = complex<double>(0.0, 0.0);
    complex<double> R_m0_10 = complex<double>(0.0, 0.0);
    complex<double> R_m0_11 = complex<double>(0.0, 0.0);
    complex<double> R_m1_00 = complex<double>(0.0, 0.0);
    complex<double> R_m1_01 = complex<double>(0.0, 0.0);
    complex<double> R_m1_10 = complex<double>(0.0, 0.0);
    complex<double> R_m1_11 = complex<double>(0.0, 0.0);

    for (int j = 0; j < freq; j++) {
        int ptcl_idx = incl_offset - 1 - j;
        double xj = x[ptcl_idx];
        double yj = y[ptcl_idx];
        double zj = z[ptcl_idx];
        double wj = q * w[ptcl_idx];

        double rj = sqrt(xj * xj + yj * yj);
        double cos = 1.0, sin = 0.0;
        if (rj != 0.0) {
            double invr = 1.0 / rj;
            cos = xj * invr;
            sin = yj * invr;
        }

        const complex<double> exptheta_m0 = 1.0;
        const complex<double> exptheta_m1 = complex<double>(cos, sin);

        double z_cell = invdz * (zj - zmin) - 0.5;
        double r_cell = invdr * (rj - rmin) - 0.5;

        int ir = min(int(ceil(r_cell)), Nr);
        double beta0 = beta_n_m0[ir];
        double beta1 = beta_n_m1[ir];

        double Sr, Sz, weight;

        // m = 0
        const complex <double> R0 = wj * exptheta_m0;
        Sr = Sr_linear(r_cell, 0, 1, beta0);
        Sz = Sz_linear(z_cell, 0);
        weight = Sr * Sz;
        R_m0_00 += weight * R0;

        Sr = Sr_linear(r_cell, 1, 1, beta0);
        Sz = Sz_linear(z_cell, 0);
        weight = Sr * Sz;
        R_m0_10 += weight * R0;

        Sr = Sr_linear(r_cell, 0, 1, beta0);
        Sz = Sz_linear(z_cell, 1);
        weight = Sr * Sz;
        R_m0_01 += weight * R0;

        Sr = Sr_linear(r_cell, 1, 1, beta0);
        Sz = Sz_linear(z_cell, 1);
        weight = Sr * Sz;
        R_m0_11 += weight * R0;

        // m = 1
        const complex<double> R1 = wj * exptheta_m1;
        Sr = Sr_linear(r_cell, 0, -1, beta1);
        Sz = Sz_linear(z_cell, 0);
        weight = Sr * Sz;
        R_m1_00 += R1 * weight;

        Sr = Sr_linear(r_cell, 1, -1, beta1);
        Sz = Sz_linear(z_cell, 0);
        weight = Sr * Sz;
        R_m1_10 += R1 * weight;

        Sr = Sr_linear(r_cell, 0, -1, beta1);
        Sz = Sz_linear(z_cell, 1);
        weight = Sr * Sz;
        R_m1_01 += R1 * weight;

        Sr = Sr_linear(r_cell, 1, -1, beta1);
        Sz = Sz_linear(z_cell, 1);
        weight = Sr * Sz;
        R_m1_11 += R1 * weight;
    }

    int iz0 = iz_upper - 1;
    if (iz0 < 0) iz0 += Nz;
    int iz1 = iz_upper;

    int ir0 = ir_upper - 1;
    if (ir0 < 0) ir0 = -(1 + ir0);
    int ir1 = (ir_upper < Nr) ? ir_upper : (Nr - 1);
    double* rho_m0_x = reinterpret_cast<double*>(rho_m0);
    double* rho_m1_x = reinterpret_cast<double*>(rho_m1);
    if (freq > 0) {
        //mode 0
        atomicAdd(&rho_m0_x[2*(iz0 * Nr + ir0)], real(R_m0_00));
        atomicAdd(&rho_m0_x[2*(iz0 * Nr + ir1)], real(R_m0_10));
        atomicAdd(&rho_m0_x[2*(iz1 * Nr + ir0)], real(R_m0_01));
        atomicAdd(&rho_m0_x[2*(iz1 * Nr + ir1)], real(R_m0_11));
        //mode 1
        atomicAdd(&rho_m1_x[2*(iz0 * Nr + ir0)], real(R_m1_00));
        atomicAdd(&rho_m1_x[2*(iz0 * Nr + ir0)+1], imag(R_m1_00));
        atomicAdd(&rho_m1_x[2*(iz0 * Nr + ir1)], real(R_m1_10));
        atomicAdd(&rho_m1_x[2*(iz0 * Nr + ir1)+1], imag(R_m1_10));
        atomicAdd(&rho_m1_x[2*(iz1 * Nr + ir0)], real(R_m1_01));
        atomicAdd(&rho_m1_x[2*(iz1 * Nr + ir0)+1], imag(R_m1_01));
        atomicAdd(&rho_m1_x[2*(iz1 * Nr + ir1)], real(R_m1_11));
        atomicAdd(&rho_m1_x[2*(iz1 * Nr + ir1)+1], imag(R_m1_11));
        //atomicAdd(&real(rho_m0[iz0 * Nr + ir1]), real(R_m0_10));
        //atomicAdd(&real(rho_m0[iz1 * Nr + ir0]), real(R_m0_01));
        //atomicAdd(&real(rho_m0[iz1 * Nr + ir1]), real(R_m0_11));
        //atomicAdd(&real(rho_m1[iz0 * Nr + ir0]), real(R_m1_00));
        //atomicAdd(&real(rho_m1[iz0 * Nr + ir1]), real(R_m1_10));
        //atomicAdd(&real(rho_m1[iz1 * Nr + ir0]), real(R_m1_01));
        //atomicAdd(&real(rho_m1[iz1 * Nr + ir1]), real(R_m1_11));
    }
}


extern "C" __global__
void deposit_J_gpu_linear(
    const double* x,
    const double* y,
    const double* z,
    const double* w,
    const double  q,
    const double* ux,
    const double* uy,
    const double* uz,
    const double* inv_gamma,
    const double  invdz,
    const double  zmin,
    const int     Nz,
    const double  invdr,
    const double  rmin,
    const int     Nr,
    complex<double>* j_r_m0,
    complex<double>* j_r_m1,
    complex<double>* j_t_m0,
    complex<double>* j_t_m1,
    complex<double>* j_z_m0,
    complex<double>* j_z_m1,
    const int* cell_idx,          // unused in this kernel but kept for signature parity
    const int* prefix_sum,
    const double* beta_n_m0,
    const double* beta_n_m1,
    const double c,
    const int   Ncells       // length of prefix_sum (number of "cells")
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Ncells) return;

    // Retrieve iz_upper, ir_upper from i (same arithmetic as python)
    int iz_upper = i / (Nr + 1);
    int ir_upper = i - iz_upper * (Nr + 1);

    // inclusive offset
    int incl_offset = prefix_sum[i];

    int frequency_per_cell;
    if (i > 0) frequency_per_cell = incl_offset - prefix_sum[i - 1];
    else frequency_per_cell = incl_offset;

    // local accumulators (mode0: real only; mode1: complex)
    double J_r_m0_00 = 0.0, J_r_m0_01 = 0.0, J_r_m0_10 = 0.0, J_r_m0_11 = 0.0;
    complex<double> J_r_m1_00(0.0,0.0), J_r_m1_01(0.0,0.0), J_r_m1_10(0.0,0.0), J_r_m1_11(0.0,0.0);

    double J_t_m0_00 = 0.0, J_t_m0_01 = 0.0, J_t_m0_10 = 0.0, J_t_m0_11 = 0.0;
    complex<double> J_t_m1_00(0.0,0.0), J_t_m1_01(0.0,0.0), J_t_m1_10(0.0,0.0), J_t_m1_11(0.0,0.0);

    double J_z_m0_00 = 0.0, J_z_m0_01 = 0.0, J_z_m0_10 = 0.0, J_z_m0_11 = 0.0;
    complex<double> J_z_m1_00(0.0,0.0), J_z_m1_01(0.0,0.0), J_z_m1_10(0.0,0.0), J_z_m1_11(0.0,0.0);

    // speed of light constant (same use as original)
    //const double c = 299792458.0;

    // Loop over particles in this cell (unrolled in python; here explicit loop)
    for (int j = 0; j < frequency_per_cell; ++j) {
        int ptcl_idx = incl_offset - 1 - j;

        // load particle data
        double xj = x[ptcl_idx];
        double yj = y[ptcl_idx];
        double zj = z[ptcl_idx];
        double uxj = ux[ptcl_idx];
        double uyj = uy[ptcl_idx];
        double uzj = uz[ptcl_idx];
        double inv_gammaj = inv_gamma[ptcl_idx];
        double wj = q * w[ptcl_idx];

        // cylindrical conversion
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

        // azimuthal factors (mode0 and mode1 as in your Numba code)
        double exptheta_m0 = 1.0;
        complex<double> exptheta_m1 = complex<double>(cosv, sinv);

        // positions in cell units
        double r_cell = invdr * (rj - rmin) - 0.5;
        double z_cell = invdz * (zj - zmin) - 0.5;

        // Ruyten-corrected shape coefficient indices
        int ir = min((int)ceil(r_cell), Nr);
        double bn0 = beta_n_m0[ir];
        double bn1 = beta_n_m1[ir];

        // compute scalars
        // Mode 0 (real)
        double Jr_m0_base = wj * c * inv_gammaj * (cosv * uxj + sinv * uyj); // real
        double Jt_m0_base = wj * c * inv_gammaj * (cosv * uyj - sinv * uxj);
        double Jz_m0_base = wj * c * inv_gammaj * uzj;

        // Mode 1 (complex, multiply by exptheta_m1)
        complex<double> Jr_m1_base = complex<double>(wj * c * inv_gammaj * (cosv * uxj + sinv * uyj), 0.0) * exptheta_m1;
        complex<double> Jt_m1_base = complex<double>(wj * c * inv_gammaj * (cosv * uyj - sinv * uxj), 0.0) * exptheta_m1;
        complex<double> Jz_m1_base = complex<double>(wj * c * inv_gammaj * uzj, 0.0) * exptheta_m1;

        // Accumulate to 4 nodes using linear shape functions (exact mapping from Numba)
        // mode 0 contributions (real)
        J_r_m0_00 += Sr_linear(r_cell, 0, -1.0, bn0) * Sz_linear(z_cell, 0) * Jr_m0_base;
        J_t_m0_00 += Sr_linear(r_cell, 0, -1.0, bn0) * Sz_linear(z_cell, 0) * Jt_m0_base;
        J_z_m0_00 += Sr_linear(r_cell, 0,  1.0, bn0) * Sz_linear(z_cell, 0) * Jz_m0_base;

        J_r_m0_01 += Sr_linear(r_cell, 0, -1.0, bn0) * Sz_linear(z_cell, 1) * Jr_m0_base;
        J_t_m0_01 += Sr_linear(r_cell, 0, -1.0, bn0) * Sz_linear(z_cell, 1) * Jt_m0_base;
        J_z_m0_01 += Sr_linear(r_cell, 0,  1.0, bn0) * Sz_linear(z_cell, 1) * Jz_m0_base;

        J_r_m0_10 += Sr_linear(r_cell, 1, -1.0, bn0) * Sz_linear(z_cell, 0) * Jr_m0_base;
        J_t_m0_10 += Sr_linear(r_cell, 1, -1.0, bn0) * Sz_linear(z_cell, 0) * Jt_m0_base;
        J_z_m0_10 += Sr_linear(r_cell, 1,  1.0, bn0) * Sz_linear(z_cell, 0) * Jz_m0_base;

        J_r_m0_11 += Sr_linear(r_cell, 1, -1.0, bn0) * Sz_linear(z_cell, 1) * Jr_m0_base;
        J_t_m0_11 += Sr_linear(r_cell, 1, -1.0, bn0) * Sz_linear(z_cell, 1) * Jt_m0_base;
        J_z_m0_11 += Sr_linear(r_cell, 1,  1.0, bn0) * Sz_linear(z_cell, 1) * Jz_m0_base;

        // mode 1 contributions (complex)
        J_r_m1_00 += Sr_linear(r_cell, 0, 1.0, bn1) * Sz_linear(z_cell, 0) * Jr_m1_base;
        J_t_m1_00 += Sr_linear(r_cell, 0, 1.0, bn1) * Sz_linear(z_cell, 0) * Jt_m1_base;
        J_z_m1_00 += Sr_linear(r_cell, 0, -1.0, bn1) * Sz_linear(z_cell, 0) * Jz_m1_base;

        J_r_m1_01 += Sr_linear(r_cell, 0, 1.0, bn1) * Sz_linear(z_cell, 1) * Jr_m1_base;
        J_t_m1_01 += Sr_linear(r_cell, 0, 1.0, bn1) * Sz_linear(z_cell, 1) * Jt_m1_base;
        J_z_m1_01 += Sr_linear(r_cell, 0, -1.0, bn1) * Sz_linear(z_cell, 1) * Jz_m1_base;

        J_r_m1_10 += Sr_linear(r_cell, 1, 1.0, bn1) * Sz_linear(z_cell, 0) * Jr_m1_base;
        J_t_m1_10 += Sr_linear(r_cell, 1, 1.0, bn1) * Sz_linear(z_cell, 0) * Jt_m1_base;
        J_z_m1_10 += Sr_linear(r_cell, 1, -1.0, bn1) * Sz_linear(z_cell, 0) * Jz_m1_base;

        J_r_m1_11 += Sr_linear(r_cell, 1, 1.0, bn1) * Sz_linear(z_cell, 1) * Jr_m1_base;
        J_t_m1_11 += Sr_linear(r_cell, 1, 1.0, bn1) * Sz_linear(z_cell, 1) * Jt_m1_base;
        J_z_m1_11 += Sr_linear(r_cell, 1, -1.0, bn1) * Sz_linear(z_cell, 1) * Jz_m1_base;
    } // end for each particle in cell

    // compute indices and linear positions
    int iz0 = iz_upper - 1;
    int iz1 = iz_upper;
    if (iz0 < 0) iz0 += Nz;

    int ir0 = ir_upper - 1;
    int ir1 = min(ir_upper, Nr - 1);
    if (ir0 < 0) ir0 = -(1 + ir0);

    int idx00 = iz0 * Nr + ir0;
    int idx10 = iz0 * Nr + ir1;
    int idx01 = iz1 * Nr + ir0;
    int idx11 = iz1 * Nr + ir1;

    if (frequency_per_cell > 0) {
        // reinterpret complex arrays as double* (real, imag interleaved)
        double* j_r_m0_x = reinterpret_cast<double*>(j_r_m0);
        double* j_r_m1_x = reinterpret_cast<double*>(j_r_m1);
        double* j_t_m0_x = reinterpret_cast<double*>(j_t_m0);
        double* j_t_m1_x = reinterpret_cast<double*>(j_t_m1);
        double* j_z_m0_x = reinterpret_cast<double*>(j_z_m0);
        double* j_z_m1_x = reinterpret_cast<double*>(j_z_m1);

        // jr mode0: only real parts
        atomicAdd(&j_r_m0_x[2 * idx00 + 0], J_r_m0_00);
        atomicAdd(&j_r_m0_x[2 * idx10 + 0], J_r_m0_10);
        atomicAdd(&j_r_m0_x[2 * idx01 + 0], J_r_m0_01);
        atomicAdd(&j_r_m0_x[2 * idx11 + 0], J_r_m0_11);

        // jr mode1: real + imag
        atomicAdd(&j_r_m1_x[2 * idx00 + 0], J_r_m1_00.real());
        atomicAdd(&j_r_m1_x[2 * idx00 + 1], J_r_m1_00.imag());
        atomicAdd(&j_r_m1_x[2 * idx10 + 0], J_r_m1_10.real());
        atomicAdd(&j_r_m1_x[2 * idx10 + 1], J_r_m1_10.imag());
        atomicAdd(&j_r_m1_x[2 * idx01 + 0], J_r_m1_01.real());
        atomicAdd(&j_r_m1_x[2 * idx01 + 1], J_r_m1_01.imag());
        atomicAdd(&j_r_m1_x[2 * idx11 + 0], J_r_m1_11.real());
        atomicAdd(&j_r_m1_x[2 * idx11 + 1], J_r_m1_11.imag());

        // jt mode0
        atomicAdd(&j_t_m0_x[2 * idx00 + 0], J_t_m0_00);
        atomicAdd(&j_t_m0_x[2 * idx10 + 0], J_t_m0_10);
        atomicAdd(&j_t_m0_x[2 * idx01 + 0], J_t_m0_01);
        atomicAdd(&j_t_m0_x[2 * idx11 + 0], J_t_m0_11);

        // jt mode1
        atomicAdd(&j_t_m1_x[2 * idx00 + 0], J_t_m1_00.real());
        atomicAdd(&j_t_m1_x[2 * idx00 + 1], J_t_m1_00.imag());
        atomicAdd(&j_t_m1_x[2 * idx10 + 0], J_t_m1_10.real());
        atomicAdd(&j_t_m1_x[2 * idx10 + 1], J_t_m1_10.imag());
        atomicAdd(&j_t_m1_x[2 * idx01 + 0], J_t_m1_01.real());
        atomicAdd(&j_t_m1_x[2 * idx01 + 1], J_t_m1_01.imag());
        atomicAdd(&j_t_m1_x[2 * idx11 + 0], J_t_m1_11.real());
        atomicAdd(&j_t_m1_x[2 * idx11 + 1], J_t_m1_11.imag());

        // jz mode0
        atomicAdd(&j_z_m0_x[2 * idx00 + 0], J_z_m0_00);
        atomicAdd(&j_z_m0_x[2 * idx10 + 0], J_z_m0_10);
        atomicAdd(&j_z_m0_x[2 * idx01 + 0], J_z_m0_01);
        atomicAdd(&j_z_m0_x[2 * idx11 + 0], J_z_m0_11);

        // jz mode1
        atomicAdd(&j_z_m1_x[2 * idx00 + 0], J_z_m1_00.real());
        atomicAdd(&j_z_m1_x[2 * idx00 + 1], J_z_m1_00.imag());
        atomicAdd(&j_z_m1_x[2 * idx10 + 0], J_z_m1_10.real());
        atomicAdd(&j_z_m1_x[2 * idx10 + 1], J_z_m1_10.imag());
        atomicAdd(&j_z_m1_x[2 * idx01 + 0], J_z_m1_01.real());
        atomicAdd(&j_z_m1_x[2 * idx01 + 1], J_z_m1_01.imag());
        atomicAdd(&j_z_m1_x[2 * idx11 + 0], J_z_m1_11.real());
        atomicAdd(&j_z_m1_x[2 * idx11 + 1], J_z_m1_11.imag());
    }
}




extern "C" __global__
void deposit_rho_gpu_cubic(
    const double* x,
    const double* y,
    const double* z,
    const double* w,
    const double  q,
    const double  invdz,
    const double  zmin,
    const int     Nz,
    const double  invdr,
    const double  rmin,
    const int     Nr,
    complex<double>* rho_m0,
    complex<double>* rho_m1,
    const int* cell_idx,
    const int* prefix_sum,
    const double* beta_n_m0,
    const double* beta_n_m1,
    const int     Ncells   // length of prefix_sum (number of cells)
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Ncells) return;

    int iz_upper = i / (Nr + 1);
    int ir_upper = i - iz_upper * (Nr + 1);

    int incl_offset = prefix_sum[i];

    int frequency_per_cell;
    if (i > 0) frequency_per_cell = incl_offset - prefix_sum[i - 1];
    else frequency_per_cell = incl_offset;

    // local registers (mode0: real; mode1: complex)
    double R_m0_00 = 0.0, R_m0_01 = 0.0, R_m0_02 = 0.0, R_m0_03 = 0.0;
    double R_m0_10 = 0.0, R_m0_11 = 0.0, R_m0_12 = 0.0, R_m0_13 = 0.0;
    double R_m0_20 = 0.0, R_m0_21 = 0.0, R_m0_22 = 0.0, R_m0_23 = 0.0;
    double R_m0_30 = 0.0, R_m0_31 = 0.0, R_m0_32 = 0.0, R_m0_33 = 0.0;

    complex<double> R_m1_00(0.0,0.0), R_m1_01(0.0,0.0), R_m1_02(0.0,0.0), R_m1_03(0.0,0.0);
    complex<double> R_m1_10(0.0,0.0), R_m1_11(0.0,0.0), R_m1_12(0.0,0.0), R_m1_13(0.0,0.0);
    complex<double> R_m1_20(0.0,0.0), R_m1_21(0.0,0.0), R_m1_22(0.0,0.0), R_m1_23(0.0,0.0);
    complex<double> R_m1_30(0.0,0.0), R_m1_31(0.0,0.0), R_m1_32(0.0,0.0), R_m1_33(0.0,0.0);

    // Sign flips as in Python code: mode0 flip=1, mode1 flip=-1 in places used
    double flip_m0 = 1.0;
    double flip_m1 = -1.0;

    for (int j = 0; j < frequency_per_cell; ++j) {
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

        // azimuthal factors
        double exptheta_m0 = 1.0;
        complex<double> exptheta_m1 = complex<double>(cosv, sinv);

        double r_cell = invdr * (rj - rmin) - 0.5;
        double z_cell = invdz * (zj - zmin) - 0.5;

        int ir = min((int)ceil(r_cell), Nr);
        double bn_m0 = beta_n_m0[ir];
        double bn_m1 = beta_n_m1[ir];

        // scalars
        double R_m0_scal = wj * exptheta_m0;                 // real
        complex<double> R_m1_scal = complex<double>(wj,0.0) * exptheta_m1; // complex

        // accumulate exactly as in Python code
        R_m0_00 += Sr_cubic(r_cell, 0, flip_m0, bn_m0) * Sz_cubic(z_cell, 0) * R_m0_scal;
        R_m1_00 += Sr_cubic(r_cell, 0, flip_m1, bn_m1) * Sz_cubic(z_cell, 0) * R_m1_scal;

        R_m0_01 += Sr_cubic(r_cell, 0, flip_m0, bn_m0) * Sz_cubic(z_cell, 1) * R_m0_scal;
        R_m1_01 += Sr_cubic(r_cell, 0, flip_m1, bn_m1) * Sz_cubic(z_cell, 1) * R_m1_scal;

        R_m0_02 += Sr_cubic(r_cell, 0, flip_m0, bn_m0) * Sz_cubic(z_cell, 2) * R_m0_scal;
        R_m1_02 += Sr_cubic(r_cell, 0, flip_m1, bn_m1) * Sz_cubic(z_cell, 2) * R_m1_scal;

        R_m0_03 += Sr_cubic(r_cell, 0, flip_m0, bn_m0) * Sz_cubic(z_cell, 3) * R_m0_scal;
        R_m1_03 += Sr_cubic(r_cell, 0, flip_m1, bn_m1) * Sz_cubic(z_cell, 3) * R_m1_scal;

        R_m0_10 += Sr_cubic(r_cell, 1, flip_m0, bn_m0) * Sz_cubic(z_cell, 0) * R_m0_scal;
        R_m1_10 += Sr_cubic(r_cell, 1, flip_m1, bn_m1) * Sz_cubic(z_cell, 0) * R_m1_scal;

        R_m0_11 += Sr_cubic(r_cell, 1, flip_m0, bn_m0) * Sz_cubic(z_cell, 1) * R_m0_scal;
        R_m1_11 += Sr_cubic(r_cell, 1, flip_m1, bn_m1) * Sz_cubic(z_cell, 1) * R_m1_scal;

        R_m0_12 += Sr_cubic(r_cell, 1, flip_m0, bn_m0) * Sz_cubic(z_cell, 2) * R_m0_scal;
        R_m1_12 += Sr_cubic(r_cell, 1, flip_m1, bn_m1) * Sz_cubic(z_cell, 2) * R_m1_scal;

        R_m0_13 += Sr_cubic(r_cell, 1, flip_m0, bn_m0) * Sz_cubic(z_cell, 3) * R_m0_scal;
        R_m1_13 += Sr_cubic(r_cell, 1, flip_m1, bn_m1) * Sz_cubic(z_cell, 3) * R_m1_scal;

        R_m0_20 += Sr_cubic(r_cell, 2, flip_m0, bn_m0) * Sz_cubic(z_cell, 0) * R_m0_scal;
        R_m1_20 += Sr_cubic(r_cell, 2, flip_m1, bn_m1) * Sz_cubic(z_cell, 0) * R_m1_scal;

        R_m0_21 += Sr_cubic(r_cell, 2, flip_m0, bn_m0) * Sz_cubic(z_cell, 1) * R_m0_scal;
        R_m1_21 += Sr_cubic(r_cell, 2, flip_m1, bn_m1) * Sz_cubic(z_cell, 1) * R_m1_scal;

        R_m0_22 += Sr_cubic(r_cell, 2, flip_m0, bn_m0) * Sz_cubic(z_cell, 2) * R_m0_scal;
        R_m1_22 += Sr_cubic(r_cell, 2, flip_m1, bn_m1) * Sz_cubic(z_cell, 2) * R_m1_scal;

        R_m0_23 += Sr_cubic(r_cell, 2, flip_m0, bn_m0) * Sz_cubic(z_cell, 3) * R_m0_scal;
        R_m1_23 += Sr_cubic(r_cell, 2, flip_m1, bn_m1) * Sz_cubic(z_cell, 3) * R_m1_scal;

        R_m0_30 += Sr_cubic(r_cell, 3, flip_m0, bn_m0) * Sz_cubic(z_cell, 0) * R_m0_scal;
        R_m1_30 += Sr_cubic(r_cell, 3, flip_m1, bn_m1) * Sz_cubic(z_cell, 0) * R_m1_scal;

        R_m0_31 += Sr_cubic(r_cell, 3, flip_m0, bn_m0) * Sz_cubic(z_cell, 1) * R_m0_scal;
        R_m1_31 += Sr_cubic(r_cell, 3, flip_m1, bn_m1) * Sz_cubic(z_cell, 1) * R_m1_scal;

        R_m0_32 += Sr_cubic(r_cell, 3, flip_m0, bn_m0) * Sz_cubic(z_cell, 2) * R_m0_scal;
        R_m1_32 += Sr_cubic(r_cell, 3, flip_m1, bn_m1) * Sz_cubic(z_cell, 2) * R_m1_scal;

        R_m0_33 += Sr_cubic(r_cell, 3, flip_m0, bn_m0) * Sz_cubic(z_cell, 3) * R_m0_scal;
        R_m1_33 += Sr_cubic(r_cell, 3, flip_m1, bn_m1) * Sz_cubic(z_cell, 3) * R_m1_scal;
    } // end for particles in cell

    // compute indices for grid positions and fold / periodic handling (same as Python)
    int iz0 = iz_upper - 2;
    int iz1 = iz_upper - 1;
    int iz2 = iz_upper;
    int iz3 = iz_upper + 1;
    if (iz0 < 0) iz0 += Nz;
    if (iz1 < 0) iz1 += Nz;
    if (iz3 > Nz - 1) iz3 -= Nz;

    int ir0 = ir_upper - 2;
    int ir1 = min(ir_upper - 1, Nr - 1);
    int ir2 = min(ir_upper, Nr - 1);
    int ir3 = min(ir_upper + 1, Nr - 1);
    if (ir0 < 0) ir0 = -(1 + ir0);
    if (ir1 < 0) ir1 = -(1 + ir1);

    if (frequency_per_cell > 0) {
        // reinterpret complex arrays as double* (real, imag interleaved)
        double* rho_m0_x = reinterpret_cast<double*>(rho_m0);
        double* rho_m1_x = reinterpret_cast<double*>(rho_m1);

        // compute flat indices for 2D (iz * Nr + ir)
        int idx_00 = iz0 * Nr + ir0;
        int idx_10 = iz0 * Nr + ir1;
        int idx_20 = iz0 * Nr + ir2;
        int idx_30 = iz0 * Nr + ir3;

        int idx_01 = iz1 * Nr + ir0;
        int idx_11 = iz1 * Nr + ir1;
        int idx_21 = iz1 * Nr + ir2;
        int idx_31 = iz1 * Nr + ir3;

        int idx_02 = iz2 * Nr + ir0;
        int idx_12 = iz2 * Nr + ir1;
        int idx_22 = iz2 * Nr + ir2;
        int idx_32 = iz2 * Nr + ir3;

        int idx_03 = iz3 * Nr + ir0;
        int idx_13 = iz3 * Nr + ir1;
        int idx_23 = iz3 * Nr + ir2;
        int idx_33 = iz3 * Nr + ir3;

        // Mode 0: only real parts (store at 2*idx + 0)
        atomicAdd(&rho_m0_x[2 * idx_00 + 0], R_m0_00);
        atomicAdd(&rho_m0_x[2 * idx_10 + 0], R_m0_10);
        atomicAdd(&rho_m0_x[2 * idx_20 + 0], R_m0_20);
        atomicAdd(&rho_m0_x[2 * idx_30 + 0], R_m0_30);

        atomicAdd(&rho_m0_x[2 * idx_01 + 0], R_m0_01);
        atomicAdd(&rho_m0_x[2 * idx_11 + 0], R_m0_11);
        atomicAdd(&rho_m0_x[2 * idx_21 + 0], R_m0_21);
        atomicAdd(&rho_m0_x[2 * idx_31 + 0], R_m0_31);

        atomicAdd(&rho_m0_x[2 * idx_02 + 0], R_m0_02);
        atomicAdd(&rho_m0_x[2 * idx_12 + 0], R_m0_12);
        atomicAdd(&rho_m0_x[2 * idx_22 + 0], R_m0_22);
        atomicAdd(&rho_m0_x[2 * idx_32 + 0], R_m0_32);

        atomicAdd(&rho_m0_x[2 * idx_03 + 0], R_m0_03);
        atomicAdd(&rho_m0_x[2 * idx_13 + 0], R_m0_13);
        atomicAdd(&rho_m0_x[2 * idx_23 + 0], R_m0_23);
        atomicAdd(&rho_m0_x[2 * idx_33 + 0], R_m0_33);

        // Mode 1: real + imag
        atomicAdd(&rho_m1_x[2 * idx_00 + 0], R_m1_00.real());
        atomicAdd(&rho_m1_x[2 * idx_00 + 1], R_m1_00.imag());
        atomicAdd(&rho_m1_x[2 * idx_10 + 0], R_m1_10.real());
        atomicAdd(&rho_m1_x[2 * idx_10 + 1], R_m1_10.imag());
        atomicAdd(&rho_m1_x[2 * idx_20 + 0], R_m1_20.real());
        atomicAdd(&rho_m1_x[2 * idx_20 + 1], R_m1_20.imag());
        atomicAdd(&rho_m1_x[2 * idx_30 + 0], R_m1_30.real());
        atomicAdd(&rho_m1_x[2 * idx_30 + 1], R_m1_30.imag());

        atomicAdd(&rho_m1_x[2 * idx_01 + 0], R_m1_01.real());
        atomicAdd(&rho_m1_x[2 * idx_01 + 1], R_m1_01.imag());
        atomicAdd(&rho_m1_x[2 * idx_11 + 0], R_m1_11.real());
        atomicAdd(&rho_m1_x[2 * idx_11 + 1], R_m1_11.imag());
        atomicAdd(&rho_m1_x[2 * idx_21 + 0], R_m1_21.real());
        atomicAdd(&rho_m1_x[2 * idx_21 + 1], R_m1_21.imag());
        atomicAdd(&rho_m1_x[2 * idx_31 + 0], R_m1_31.real());
        atomicAdd(&rho_m1_x[2 * idx_31 + 1], R_m1_31.imag());

        atomicAdd(&rho_m1_x[2 * idx_02 + 0], R_m1_02.real());
        atomicAdd(&rho_m1_x[2 * idx_02 + 1], R_m1_02.imag());
        atomicAdd(&rho_m1_x[2 * idx_12 + 0], R_m1_12.real());
        atomicAdd(&rho_m1_x[2 * idx_12 + 1], R_m1_12.imag());
        atomicAdd(&rho_m1_x[2 * idx_22 + 0], R_m1_22.real());
        atomicAdd(&rho_m1_x[2 * idx_22 + 1], R_m1_22.imag());
        atomicAdd(&rho_m1_x[2 * idx_32 + 0], R_m1_32.real());
        atomicAdd(&rho_m1_x[2 * idx_32 + 1], R_m1_32.imag());

        atomicAdd(&rho_m1_x[2 * idx_03 + 0], R_m1_03.real());
        atomicAdd(&rho_m1_x[2 * idx_03 + 1], R_m1_03.imag());
        atomicAdd(&rho_m1_x[2 * idx_13 + 0], R_m1_13.real());
        atomicAdd(&rho_m1_x[2 * idx_13 + 1], R_m1_13.imag());
        atomicAdd(&rho_m1_x[2 * idx_23 + 0], R_m1_23.real());
        atomicAdd(&rho_m1_x[2 * idx_23 + 1], R_m1_23.imag());
        atomicAdd(&rho_m1_x[2 * idx_33 + 0], R_m1_33.real());
        atomicAdd(&rho_m1_x[2 * idx_33 + 1], R_m1_33.imag());
    }
}


extern "C" __global__
void deposit_J_gpu_cubic(
    const double* x,
    const double* y,
    const double* z,
    const double* w,
    const double  q,
    const double* ux,
    const double* uy,
    const double* uz,
    const double* inv_gamma,
    const double  invdz,
    const double  zmin,
    const int     Nz,
    const double  invdr,
    const double  rmin,
    const int     Nr,               
    complex<double>* j_r_m0,      // mode 0 real arrays (Nz x Nr) flattened
    complex<double>* j_r_m1,      // mode 1 complex arrays (Nz x Nr) flattened
    complex<double>* j_t_m0,
    complex<double>* j_t_m1,
    complex<double>* j_z_m0,
    complex<double>* j_z_m1,
    const int* cell_idx,
    const int* prefix_sum,
    const double* beta_n_m0,
    const double* beta_n_m1,
    const double c,
    const int prefix_len        // prefix_sum length (#cells)
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= prefix_len) return;

    //const double c = 299792458.0;

    // compute iz_upper and ir_upper like python
    int iz_upper = i / (Nr + 1);
    int ir_upper = i - iz_upper * (Nr + 1);

    int incl_offset = prefix_sum[i];

    int frequency_per_cell;
    if (i > 0) frequency_per_cell = incl_offset - prefix_sum[i - 1];
    else frequency_per_cell = incl_offset;

    // --- local accumulators ---
    // mode0 are real scalars
    double J_r_m0_00=0., J_t_m0_00=0., J_z_m0_00=0.;
    double J_r_m0_01=0., J_t_m0_01=0., J_z_m0_01=0.;
    double J_r_m0_02=0., J_t_m0_02=0., J_z_m0_02=0.;
    double J_r_m0_03=0., J_t_m0_03=0., J_z_m0_03=0.;
    double J_r_m0_10=0., J_t_m0_10=0., J_z_m0_10=0.;
    double J_r_m0_11=0., J_t_m0_11=0., J_z_m0_11=0.;
    double J_r_m0_12=0., J_t_m0_12=0., J_z_m0_12=0.;
    double J_r_m0_13=0., J_t_m0_13=0., J_z_m0_13=0.;
    double J_r_m0_20=0., J_t_m0_20=0., J_z_m0_20=0.;
    double J_r_m0_21=0., J_t_m0_21=0., J_z_m0_21=0.;
    double J_r_m0_22=0., J_t_m0_22=0., J_z_m0_22=0.;
    double J_r_m0_23=0., J_t_m0_23=0., J_z_m0_23=0.;
    double J_r_m0_30=0., J_t_m0_30=0., J_z_m0_30=0.;
    double J_r_m0_31=0., J_t_m0_31=0., J_z_m0_31=0.;
    double J_r_m0_32=0., J_t_m0_32=0., J_z_m0_32=0.;
    double J_r_m0_33=0., J_t_m0_33=0., J_z_m0_33=0.;

    // mode1 are complex
    complex<double> J_r_m1_00(0.,0.), J_t_m1_00(0.,0.), J_z_m1_00(0.,0.);
    complex<double> J_r_m1_01(0.,0.), J_t_m1_01(0.,0.), J_z_m1_01(0.,0.);
    complex<double> J_r_m1_02(0.,0.), J_t_m1_02(0.,0.), J_z_m1_02(0.,0.);
    complex<double> J_r_m1_03(0.,0.), J_t_m1_03(0.,0.), J_z_m1_03(0.,0.);
    complex<double> J_r_m1_10(0.,0.), J_t_m1_10(0.,0.), J_z_m1_10(0.,0.);
    complex<double> J_r_m1_11(0.,0.), J_t_m1_11(0.,0.), J_z_m1_11(0.,0.);
    complex<double> J_r_m1_12(0.,0.), J_t_m1_12(0.,0.), J_z_m1_12(0.,0.);
    complex<double> J_r_m1_13(0.,0.), J_t_m1_13(0.,0.), J_z_m1_13(0.,0.);
    complex<double> J_r_m1_20(0.,0.), J_t_m1_20(0.,0.), J_z_m1_20(0.,0.);
    complex<double> J_r_m1_21(0.,0.), J_t_m1_21(0.,0.), J_z_m1_21(0.,0.);
    complex<double> J_r_m1_22(0.,0.), J_t_m1_22(0.,0.), J_z_m1_22(0.,0.);
    complex<double> J_r_m1_23(0.,0.), J_t_m1_23(0.,0.), J_z_m1_23(0.,0.);
    complex<double> J_r_m1_30(0.,0.), J_t_m1_30(0.,0.), J_z_m1_30(0.,0.);
    complex<double> J_r_m1_31(0.,0.), J_t_m1_31(0.,0.), J_z_m1_31(0.,0.);
    complex<double> J_r_m1_32(0.,0.), J_t_m1_32(0.,0.), J_z_m1_32(0.,0.);
    complex<double> J_r_m1_33(0.,0.), J_t_m1_33(0.,0.), J_z_m1_33(0.,0.);

    // sign flip f = (-1)^m used in python; here will be passed implicitly by which Sr call uses what sign
    // but since we evaluate both modes separately we will pass flip = -1 or +1 accordingly where needed.

    for (int j = 0; j < frequency_per_cell; ++j) {
        int ptcl_idx = incl_offset - 1 - j;

        // positions
        double xj = x[ptcl_idx];
        double yj = y[ptcl_idx];
        double zj = z[ptcl_idx];
        // velocities
        double uxj = ux[ptcl_idx];
        double uyj = uy[ptcl_idx];
        double uzj = uz[ptcl_idx];
        double inv_gammaj = inv_gamma[ptcl_idx];
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

        // exptheta for m=0 is 1, for m=1 is (cos + i sin), for general m multiply accordingly
        // We must get 'm' from the arrays of beta? Actually original python receives m? In this kernel original signature had full m? 
        // But this function calculates mode0 (m=0) and mode1 (m=1) independently:
        complex<double> exptheta_m0(1.0, 0.0);
        complex<double> exptheta_m1(cosv, sinv);

        // Positions in cell units
        double r_cell = invdr * (rj - rmin) - 0.5;
        double z_cell = invdz * (zj - zmin) - 0.5;

        int ir = min((int)ceil(r_cell), Nr);
        double bn_m0 = beta_n_m0[ir];
        double bn_m1 = beta_n_m1[ir];

        // Scale factors (mode0 real scalar, mode1 complex scalar)
        double J_r_m0_scal = wj * c * inv_gammaj * (cosv*uxj + sinv*uyj); // real
        double J_t_m0_scal = wj * c * inv_gammaj * (cosv*uyj - sinv*uxj);
        double J_z_m0_scal = wj * c * inv_gammaj * uzj;

        complex<double> J_r_m1_scal = complex<double>(wj * c * inv_gammaj * (cosv*uxj + sinv*uyj), 0.0) * exptheta_m1;
        complex<double> J_t_m1_scal = complex<double>(wj * c * inv_gammaj * (cosv*uyj - sinv*uxj), 0.0) * exptheta_m1;
        complex<double> J_z_m1_scal = complex<double>(wj * c * inv_gammaj * uzj, 0.0) * exptheta_m1;

        // For mode0 Sr calls often used with -1 factor in python (f = (-1)**m where m=0 -> f=1, m=1 -> f=-1).
        // In the original python for j_r_m0 they used Sr_cubic(r_cell, idx, -f, bn_m0) with f = (-1)**m (m passed to function)
        // Here we follow the same calls found in the Python function you supplied:
        // (I preserved the exact same signs as in your python code earlier.)

        // J_r
        J_r_m0_00 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_r_m0_scal * 1.0); // mode0 exptheta =1
        J_r_m1_00 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_r_m1_scal;

        J_r_m0_01 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_r_m0_scal * 1.0);
        J_r_m1_01 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_r_m1_scal;

        J_r_m0_02 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_r_m0_scal * 1.0);
        J_r_m1_02 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_r_m1_scal;

        J_r_m0_03 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_r_m0_scal * 1.0);
        J_r_m1_03 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_r_m1_scal;

        J_r_m0_10 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_r_m0_scal * 1.0);
        J_r_m1_10 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_r_m1_scal;

        J_r_m0_11 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_r_m0_scal * 1.0);
        J_r_m1_11 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_r_m1_scal;

        J_r_m0_12 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_r_m0_scal * 1.0);
        J_r_m1_12 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_r_m1_scal;

        J_r_m0_13 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_r_m0_scal * 1.0);
        J_r_m1_13 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_r_m1_scal;

        J_r_m0_20 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_r_m0_scal * 1.0);
        J_r_m1_20 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_r_m1_scal;

        J_r_m0_21 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_r_m0_scal * 1.0);
        J_r_m1_21 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_r_m1_scal;

        J_r_m0_22 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_r_m0_scal * 1.0);
        J_r_m1_22 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_r_m1_scal;

        J_r_m0_23 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_r_m0_scal * 1.0);
        J_r_m1_23 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_r_m1_scal;

        J_r_m0_30 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_r_m0_scal * 1.0);
        J_r_m1_30 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_r_m1_scal;

        J_r_m0_31 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_r_m0_scal * 1.0);
        J_r_m1_31 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_r_m1_scal;

        J_r_m0_32 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_r_m0_scal * 1.0);
        J_r_m1_32 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_r_m1_scal;

        J_r_m0_33 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_r_m0_scal * 1.0);
        J_r_m1_33 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_r_m1_scal;

        // J_t (analogous)
        J_t_m0_00 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_t_m0_scal * 1.0);
        J_t_m1_00 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_t_m1_scal;
        J_t_m0_01 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_t_m0_scal * 1.0);
        J_t_m1_01 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_t_m1_scal;
        J_t_m0_02 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_t_m0_scal * 1.0);
        J_t_m1_02 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_t_m1_scal;
        J_t_m0_03 += Sr_cubic(r_cell, 0, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_t_m0_scal * 1.0);
        J_t_m1_03 += Sr_cubic(r_cell, 0,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_t_m1_scal;

        J_t_m0_10 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_t_m0_scal * 1.0);
        J_t_m1_10 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_t_m1_scal;
        J_t_m0_11 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_t_m0_scal * 1.0);
        J_t_m1_11 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_t_m1_scal;
        J_t_m0_12 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_t_m0_scal * 1.0);
        J_t_m1_12 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_t_m1_scal;
        J_t_m0_13 += Sr_cubic(r_cell, 1, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_t_m0_scal * 1.0);
        J_t_m1_13 += Sr_cubic(r_cell, 1,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_t_m1_scal;

        J_t_m0_20 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_t_m0_scal * 1.0);
        J_t_m1_20 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_t_m1_scal;
        J_t_m0_21 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_t_m0_scal * 1.0);
        J_t_m1_21 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_t_m1_scal;
        J_t_m0_22 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_t_m0_scal * 1.0);
        J_t_m1_22 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_t_m1_scal;
        J_t_m0_23 += Sr_cubic(r_cell, 2, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_t_m0_scal * 1.0);
        J_t_m1_23 += Sr_cubic(r_cell, 2,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_t_m1_scal;

        J_t_m0_30 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_t_m0_scal * 1.0);
        J_t_m1_30 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_t_m1_scal;
        J_t_m0_31 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_t_m0_scal * 1.0);
        J_t_m1_31 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_t_m1_scal;
        J_t_m0_32 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_t_m0_scal * 1.0);
        J_t_m1_32 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_t_m1_scal;
        J_t_m0_33 += Sr_cubic(r_cell, 3, -1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_t_m0_scal * 1.0);
        J_t_m1_33 += Sr_cubic(r_cell, 3,  1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_t_m1_scal;

        // J_z
        J_z_m0_00 += Sr_cubic(r_cell, 0,  1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_z_m0_scal * 1.0);
        J_z_m1_00 += Sr_cubic(r_cell, 0, -1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_z_m1_scal;
        J_z_m0_01 += Sr_cubic(r_cell, 0,  1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_z_m0_scal * 1.0);
        J_z_m1_01 += Sr_cubic(r_cell, 0, -1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_z_m1_scal;
        J_z_m0_02 += Sr_cubic(r_cell, 0,  1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_z_m0_scal * 1.0);
        J_z_m1_02 += Sr_cubic(r_cell, 0, -1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_z_m1_scal;
        J_z_m0_03 += Sr_cubic(r_cell, 0,  1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_z_m0_scal * 1.0);
        J_z_m1_03 += Sr_cubic(r_cell, 0, -1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_z_m1_scal;

        J_z_m0_10 += Sr_cubic(r_cell, 1,  1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_z_m0_scal * 1.0);
        J_z_m1_10 += Sr_cubic(r_cell, 1, -1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_z_m1_scal;
        J_z_m0_11 += Sr_cubic(r_cell, 1,  1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_z_m0_scal * 1.0);
        J_z_m1_11 += Sr_cubic(r_cell, 1, -1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_z_m1_scal;
        J_z_m0_12 += Sr_cubic(r_cell, 1,  1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_z_m0_scal * 1.0);
        J_z_m1_12 += Sr_cubic(r_cell, 1, -1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_z_m1_scal;
        J_z_m0_13 += Sr_cubic(r_cell, 1,  1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_z_m0_scal * 1.0);
        J_z_m1_13 += Sr_cubic(r_cell, 1, -1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_z_m1_scal;

        J_z_m0_20 += Sr_cubic(r_cell, 2,  1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_z_m0_scal * 1.0);
        J_z_m1_20 += Sr_cubic(r_cell, 2, -1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_z_m1_scal;
        J_z_m0_21 += Sr_cubic(r_cell, 2,  1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_z_m0_scal * 1.0);
        J_z_m1_21 += Sr_cubic(r_cell, 2, -1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_z_m1_scal;
        J_z_m0_22 += Sr_cubic(r_cell, 2,  1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_z_m0_scal * 1.0);
        J_z_m1_22 += Sr_cubic(r_cell, 2, -1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_z_m1_scal;
        J_z_m0_23 += Sr_cubic(r_cell, 2,  1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_z_m0_scal * 1.0);
        J_z_m1_23 += Sr_cubic(r_cell, 2, -1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_z_m1_scal;

        J_z_m0_30 += Sr_cubic(r_cell, 3,  1.0, bn_m0)*Sz_cubic(z_cell, 0) * (J_z_m0_scal * 1.0);
        J_z_m1_30 += Sr_cubic(r_cell, 3, -1.0, bn_m1)*Sz_cubic(z_cell, 0) * J_z_m1_scal;
        J_z_m0_31 += Sr_cubic(r_cell, 3,  1.0, bn_m0)*Sz_cubic(z_cell, 1) * (J_z_m0_scal * 1.0);
        J_z_m1_31 += Sr_cubic(r_cell, 3, -1.0, bn_m1)*Sz_cubic(z_cell, 1) * J_z_m1_scal;
        J_z_m0_32 += Sr_cubic(r_cell, 3,  1.0, bn_m0)*Sz_cubic(z_cell, 2) * (J_z_m0_scal * 1.0);
        J_z_m1_32 += Sr_cubic(r_cell, 3, -1.0, bn_m1)*Sz_cubic(z_cell, 2) * J_z_m1_scal;
        J_z_m0_33 += Sr_cubic(r_cell, 3,  1.0, bn_m0)*Sz_cubic(z_cell, 3) * (J_z_m0_scal * 1.0);
        J_z_m1_33 += Sr_cubic(r_cell, 3, -1.0, bn_m1)*Sz_cubic(z_cell, 3) * J_z_m1_scal;
    } // end particles in cell loop

    // compute indices and folding/periodic as in python
    int iz0 = iz_upper - 2;
    int iz1 = iz_upper - 1;
    int iz2 = iz_upper;
    int iz3 = iz_upper + 1;
    if (iz0 < 0) iz0 += Nz;
    if (iz1 < 0) iz1 += Nz;
    if (iz3 > Nz - 1) iz3 -= Nz;

    int ir0 = ir_upper - 2;
    int ir1 = min(ir_upper - 1, Nr - 1);
    int ir2 = min(ir_upper, Nr - 1);
    int ir3 = min(ir_upper + 1, Nr - 1);
    if (ir0 < 0) ir0 = -(1 + ir0);
    if (ir1 < 0) ir1 = -(1 + ir1);

    if (frequency_per_cell > 0) {
        // flatten indices: idx = iz * Nr + ir
        int idx_00 = iz0 * Nr + ir0;
        int idx_10 = iz0 * Nr + ir1;
        int idx_20 = iz0 * Nr + ir2;
        int idx_30 = iz0 * Nr + ir3;

        int idx_01 = iz1 * Nr + ir0;
        int idx_11 = iz1 * Nr + ir1;
        int idx_21 = iz1 * Nr + ir2;
        int idx_31 = iz1 * Nr + ir3;

        int idx_02 = iz2 * Nr + ir0;
        int idx_12 = iz2 * Nr + ir1;
        int idx_22 = iz2 * Nr + ir2;
        int idx_32 = iz2 * Nr + ir3;

        int idx_03 = iz3 * Nr + ir0;
        int idx_13 = iz3 * Nr + ir1;
        int idx_23 = iz3 * Nr + ir2;
        int idx_33 = iz3 * Nr + ir3;

        // --- atomic adds ---
        // jr mode0 (real arrays)
        double* j_r_m0_x = reinterpret_cast<double*>(j_r_m0);
        atomicAdd(&j_r_m0_x[2 *idx_00], J_r_m0_00);
        atomicAdd(&j_r_m0_x[2 *idx_10], J_r_m0_10);
        atomicAdd(&j_r_m0_x[2 *idx_20], J_r_m0_20);
        atomicAdd(&j_r_m0_x[2 *idx_30], J_r_m0_30);

        atomicAdd(&j_r_m0_x[2 *idx_01], J_r_m0_01);
        atomicAdd(&j_r_m0_x[2 *idx_11], J_r_m0_11);
        atomicAdd(&j_r_m0_x[2 *idx_21], J_r_m0_21);
        atomicAdd(&j_r_m0_x[2 *idx_31], J_r_m0_31);

        atomicAdd(&j_r_m0_x[2 *idx_02], J_r_m0_02);
        atomicAdd(&j_r_m0_x[2 *idx_12], J_r_m0_12);
        atomicAdd(&j_r_m0_x[2 *idx_22], J_r_m0_22);
        atomicAdd(&j_r_m0_x[2 *idx_32], J_r_m0_32);

        atomicAdd(&j_r_m0_x[2 *idx_03], J_r_m0_03);
        atomicAdd(&j_r_m0_x[2 *idx_13], J_r_m0_13);
        atomicAdd(&j_r_m0_x[2 *idx_23], J_r_m0_23);
        atomicAdd(&j_r_m0_x[2 *idx_33], J_r_m0_33);

        // jr mode1 (complex): reinterpret and atomic add real & imag
        double* j_r_m1_x = reinterpret_cast<double*>(j_r_m1);
        atomicAdd(&j_r_m1_x[2 * idx_00 + 0], J_r_m1_00.real());
        atomicAdd(&j_r_m1_x[2 * idx_00 + 1], J_r_m1_00.imag());
        atomicAdd(&j_r_m1_x[2 * idx_10 + 0], J_r_m1_10.real());
        atomicAdd(&j_r_m1_x[2 * idx_10 + 1], J_r_m1_10.imag());
        atomicAdd(&j_r_m1_x[2 * idx_20 + 0], J_r_m1_20.real());
        atomicAdd(&j_r_m1_x[2 * idx_20 + 1], J_r_m1_20.imag());
        atomicAdd(&j_r_m1_x[2 * idx_30 + 0], J_r_m1_30.real());
        atomicAdd(&j_r_m1_x[2 * idx_30 + 1], J_r_m1_30.imag());

        atomicAdd(&j_r_m1_x[2 * idx_01 + 0], J_r_m1_01.real());
        atomicAdd(&j_r_m1_x[2 * idx_01 + 1], J_r_m1_01.imag());
        atomicAdd(&j_r_m1_x[2 * idx_11 + 0], J_r_m1_11.real());
        atomicAdd(&j_r_m1_x[2 * idx_11 + 1], J_r_m1_11.imag());
        atomicAdd(&j_r_m1_x[2 * idx_21 + 0], J_r_m1_21.real());
        atomicAdd(&j_r_m1_x[2 * idx_21 + 1], J_r_m1_21.imag());
        atomicAdd(&j_r_m1_x[2 * idx_31 + 0], J_r_m1_31.real());
        atomicAdd(&j_r_m1_x[2 * idx_31 + 1], J_r_m1_31.imag());

        atomicAdd(&j_r_m1_x[2 * idx_02 + 0], J_r_m1_02.real());
        atomicAdd(&j_r_m1_x[2 * idx_02 + 1], J_r_m1_02.imag());
        atomicAdd(&j_r_m1_x[2 * idx_12 + 0], J_r_m1_12.real());
        atomicAdd(&j_r_m1_x[2 * idx_12 + 1], J_r_m1_12.imag());
        atomicAdd(&j_r_m1_x[2 * idx_22 + 0], J_r_m1_22.real());
        atomicAdd(&j_r_m1_x[2 * idx_22 + 1], J_r_m1_22.imag());
        atomicAdd(&j_r_m1_x[2 * idx_32 + 0], J_r_m1_32.real());
        atomicAdd(&j_r_m1_x[2 * idx_32 + 1], J_r_m1_32.imag());

        atomicAdd(&j_r_m1_x[2 * idx_03 + 0], J_r_m1_03.real());
        atomicAdd(&j_r_m1_x[2 * idx_03 + 1], J_r_m1_03.imag());
        atomicAdd(&j_r_m1_x[2 * idx_13 + 0], J_r_m1_13.real());
        atomicAdd(&j_r_m1_x[2 * idx_13 + 1], J_r_m1_13.imag());
        atomicAdd(&j_r_m1_x[2 * idx_23 + 0], J_r_m1_23.real());
        atomicAdd(&j_r_m1_x[2 * idx_23 + 1], J_r_m1_23.imag());
        atomicAdd(&j_r_m1_x[2 * idx_33 + 0], J_r_m1_33.real());
        atomicAdd(&j_r_m1_x[2 * idx_33 + 1], J_r_m1_33.imag());

        // --- jt (mode0 real, mode1 complex) ---
        double* j_t_m0_x = reinterpret_cast<double*>(j_t_m0);
        atomicAdd(&j_t_m0_x[2 * idx_00], J_t_m0_00);
        atomicAdd(&j_t_m0_x[2 * idx_10], J_t_m0_10);
        atomicAdd(&j_t_m0_x[2 * idx_20], J_t_m0_20);
        atomicAdd(&j_t_m0_x[2 * idx_30], J_t_m0_30);

        atomicAdd(&j_t_m0_x[2 * idx_01], J_t_m0_01);
        atomicAdd(&j_t_m0_x[2 * idx_11], J_t_m0_11);
        atomicAdd(&j_t_m0_x[2 * idx_21], J_t_m0_21);
        atomicAdd(&j_t_m0_x[2 * idx_31], J_t_m0_31);

        atomicAdd(&j_t_m0_x[2 * idx_02], J_t_m0_02);
        atomicAdd(&j_t_m0_x[2 * idx_12], J_t_m0_12);
        atomicAdd(&j_t_m0_x[2 * idx_22], J_t_m0_22);
        atomicAdd(&j_t_m0_x[2 * idx_32], J_t_m0_32);

        atomicAdd(&j_t_m0_x[2 * idx_03], J_t_m0_03);
        atomicAdd(&j_t_m0_x[2 * idx_13], J_t_m0_13);
        atomicAdd(&j_t_m0_x[2 * idx_23], J_t_m0_23);
        atomicAdd(&j_t_m0_x[2 * idx_33], J_t_m0_33);

        double* j_t_m1_x = reinterpret_cast<double*>(j_t_m1);
        atomicAdd(&j_t_m1_x[2 * idx_00 + 0], J_t_m1_00.real());
        atomicAdd(&j_t_m1_x[2 * idx_00 + 1], J_t_m1_00.imag());
        atomicAdd(&j_t_m1_x[2 * idx_10 + 0], J_t_m1_10.real());
        atomicAdd(&j_t_m1_x[2 * idx_10 + 1], J_t_m1_10.imag());
        atomicAdd(&j_t_m1_x[2 * idx_20 + 0], J_t_m1_20.real());
        atomicAdd(&j_t_m1_x[2 * idx_20 + 1], J_t_m1_20.imag());
        atomicAdd(&j_t_m1_x[2 * idx_30 + 0], J_t_m1_30.real());
        atomicAdd(&j_t_m1_x[2 * idx_30 + 1], J_t_m1_30.imag());

        atomicAdd(&j_t_m1_x[2 * idx_01 + 0], J_t_m1_01.real());
        atomicAdd(&j_t_m1_x[2 * idx_01 + 1], J_t_m1_01.imag());
        atomicAdd(&j_t_m1_x[2 * idx_11 + 0], J_t_m1_11.real());
        atomicAdd(&j_t_m1_x[2 * idx_11 + 1], J_t_m1_11.imag());
        atomicAdd(&j_t_m1_x[2 * idx_21 + 0], J_t_m1_21.real());
        atomicAdd(&j_t_m1_x[2 * idx_21 + 1], J_t_m1_21.imag());
        atomicAdd(&j_t_m1_x[2 * idx_31 + 0], J_t_m1_31.real());
        atomicAdd(&j_t_m1_x[2 * idx_31 + 1], J_t_m1_31.imag());

        atomicAdd(&j_t_m1_x[2 * idx_02 + 0], J_t_m1_02.real());
        atomicAdd(&j_t_m1_x[2 * idx_02 + 1], J_t_m1_02.imag());
        atomicAdd(&j_t_m1_x[2 * idx_12 + 0], J_t_m1_12.real());
        atomicAdd(&j_t_m1_x[2 * idx_12 + 1], J_t_m1_12.imag());
        atomicAdd(&j_t_m1_x[2 * idx_22 + 0], J_t_m1_22.real());
        atomicAdd(&j_t_m1_x[2 * idx_22 + 1], J_t_m1_22.imag());
        atomicAdd(&j_t_m1_x[2 * idx_32 + 0], J_t_m1_32.real());
        atomicAdd(&j_t_m1_x[2 * idx_32 + 1], J_t_m1_32.imag());

        atomicAdd(&j_t_m1_x[2 * idx_03 + 0], J_t_m1_03.real());
        atomicAdd(&j_t_m1_x[2 * idx_03 + 1], J_t_m1_03.imag());
        atomicAdd(&j_t_m1_x[2 * idx_13 + 0], J_t_m1_13.real());
        atomicAdd(&j_t_m1_x[2 * idx_13 + 1], J_t_m1_13.imag());
        atomicAdd(&j_t_m1_x[2 * idx_23 + 0], J_t_m1_23.real());
        atomicAdd(&j_t_m1_x[2 * idx_23 + 1], J_t_m1_23.imag());
        atomicAdd(&j_t_m1_x[2 * idx_33 + 0], J_t_m1_33.real());
        atomicAdd(&j_t_m1_x[2 * idx_33 + 1], J_t_m1_33.imag());

        // --- jz ---
        double* j_z_m0_x = reinterpret_cast<double*>(j_z_m0);
        atomicAdd(&j_z_m0_x[2 * idx_00], J_z_m0_00);
        atomicAdd(&j_z_m0_x[2 * idx_10], J_z_m0_10);
        atomicAdd(&j_z_m0_x[2 * idx_20], J_z_m0_20);
        atomicAdd(&j_z_m0_x[2 * idx_30], J_z_m0_30);

        atomicAdd(&j_z_m0_x[2 * idx_01], J_z_m0_01);
        atomicAdd(&j_z_m0_x[2 * idx_11], J_z_m0_11);
        atomicAdd(&j_z_m0_x[2 * idx_21], J_z_m0_21);
        atomicAdd(&j_z_m0_x[2 * idx_31], J_z_m0_31);

        atomicAdd(&j_z_m0_x[2 * idx_02], J_z_m0_02);
        atomicAdd(&j_z_m0_x[2 * idx_12], J_z_m0_12);
        atomicAdd(&j_z_m0_x[2 * idx_22], J_z_m0_22);
        atomicAdd(&j_z_m0_x[2 * idx_32], J_z_m0_32);

        atomicAdd(&j_z_m0_x[2 * idx_03], J_z_m0_03);
        atomicAdd(&j_z_m0_x[2 * idx_13], J_z_m0_13);
        atomicAdd(&j_z_m0_x[2 * idx_23], J_z_m0_23);
        atomicAdd(&j_z_m0_x[2 * idx_33], J_z_m0_33);

        double* j_z_m1_x = reinterpret_cast<double*>(j_z_m1);
        atomicAdd(&j_z_m1_x[2 * idx_00 + 0], J_z_m1_00.real());
        atomicAdd(&j_z_m1_x[2 * idx_00 + 1], J_z_m1_00.imag());
        atomicAdd(&j_z_m1_x[2 * idx_10 + 0], J_z_m1_10.real());
        atomicAdd(&j_z_m1_x[2 * idx_10 + 1], J_z_m1_10.imag());
        atomicAdd(&j_z_m1_x[2 * idx_20 + 0], J_z_m1_20.real());
        atomicAdd(&j_z_m1_x[2 * idx_20 + 1], J_z_m1_20.imag());
        atomicAdd(&j_z_m1_x[2 * idx_30 + 0], J_z_m1_30.real());
        atomicAdd(&j_z_m1_x[2 * idx_30 + 1], J_z_m1_30.imag());

        atomicAdd(&j_z_m1_x[2 * idx_01 + 0], J_z_m1_01.real());
        atomicAdd(&j_z_m1_x[2 * idx_01 + 1], J_z_m1_01.imag());
        atomicAdd(&j_z_m1_x[2 * idx_11 + 0], J_z_m1_11.real());
        atomicAdd(&j_z_m1_x[2 * idx_11 + 1], J_z_m1_11.imag());
        atomicAdd(&j_z_m1_x[2 * idx_21 + 0], J_z_m1_21.real());
        atomicAdd(&j_z_m1_x[2 * idx_21 + 1], J_z_m1_21.imag());
        atomicAdd(&j_z_m1_x[2 * idx_31 + 0], J_z_m1_31.real());
        atomicAdd(&j_z_m1_x[2 * idx_31 + 1], J_z_m1_31.imag());

        atomicAdd(&j_z_m1_x[2 * idx_02 + 0], J_z_m1_02.real());
        atomicAdd(&j_z_m1_x[2 * idx_02 + 1], J_z_m1_02.imag());
        atomicAdd(&j_z_m1_x[2 * idx_12 + 0], J_z_m1_12.real());
        atomicAdd(&j_z_m1_x[2 * idx_12 + 1], J_z_m1_12.imag());
        atomicAdd(&j_z_m1_x[2 * idx_22 + 0], J_z_m1_22.real());
        atomicAdd(&j_z_m1_x[2 * idx_22 + 1], J_z_m1_22.imag());
        atomicAdd(&j_z_m1_x[2 * idx_32 + 0], J_z_m1_32.real());
        atomicAdd(&j_z_m1_x[2 * idx_32 + 1], J_z_m1_32.imag());

        atomicAdd(&j_z_m1_x[2 * idx_03 + 0], J_z_m1_03.real());
        atomicAdd(&j_z_m1_x[2 * idx_03 + 1], J_z_m1_03.imag());
        atomicAdd(&j_z_m1_x[2 * idx_13 + 0], J_z_m1_13.real());
        atomicAdd(&j_z_m1_x[2 * idx_13 + 1], J_z_m1_13.imag());
        atomicAdd(&j_z_m1_x[2 * idx_23 + 0], J_z_m1_23.real());
        atomicAdd(&j_z_m1_x[2 * idx_23 + 1], J_z_m1_23.imag());
        atomicAdd(&j_z_m1_x[2 * idx_33 + 0], J_z_m1_33.real());
        atomicAdd(&j_z_m1_x[2 * idx_33 + 1], J_z_m1_33.imag());
    } // end frequency_per_cell > 0
}

