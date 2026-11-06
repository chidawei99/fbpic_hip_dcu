#include <cupy/complex.cuh>
// File-scope constants (embedded numeric literals)
//static __device__ __constant__ double C2 = {_C2:.20e};
//static __device__ __constant__ double EPS0 = {_EPS0:.20e};
//static __device__ __constant__ double MU0 = {_MU0:.20e};

// 1. Erase scalar
extern "C" __global__
void cuda_erase_scalar_kernel(
    complex<double>* __restrict__ array,
    const int Nz, const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        array[idx] = complex<double>(0.0, 0.0);
    }
}

// 2. Erase vector
extern "C" __global__
void cuda_erase_vector_kernel(
    complex<double>* __restrict__ array_r,
    complex<double>* __restrict__ array_t,
    complex<double>* __restrict__ array_z,
    const int Nz, const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        array_r[idx] = complex<double>(0.0, 0.0);
        array_t[idx] = complex<double>(0.0, 0.0);
        array_z[idx] = complex<double>(0.0, 0.0);
    }
}

// 3. Divide scalar by volume
extern "C" __global__
void cuda_divide_scalar_by_volume_kernel(
    complex<double>* __restrict__ array,
    const double* __restrict__ invvol,
    const int Nz, const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        array[idx] = array[idx] * invvol[ir];
    }
}

// 4. Divide vector by volume
extern "C" __global__
void cuda_divide_vector_by_volume_kernel(
    complex<double>* __restrict__ array_r,
    complex<double>* __restrict__ array_t,
    complex<double>* __restrict__ array_z,
    const double* __restrict__ invvol,
    const int Nz, const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        double v = invvol[ir];
        array_r[idx] = array_r[idx] * v;
        array_t[idx] = array_t[idx] * v;
        array_z[idx] = array_z[idx] * v;
    }
}

// 5. Correct currents (curl-free standard)
extern "C" __global__
void cuda_correct_currents_curlfree_standard_kernel(
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    complex<double>* __restrict__ Jp,
    complex<double>* __restrict__ Jm,
    complex<double>* __restrict__ Jz,
    const double* __restrict__ kz,
    const double* __restrict__ kr,
    const double* __restrict__ inv_k2,
    const double inv_dt,
    const int Nz, const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        complex<double> F =
            -inv_k2[idx] * (
                (rho_next[idx] - rho_prev[idx]) * inv_dt
                + complex<double>(0.0,1.0) * kz[idx] * Jz[idx]
                + kr[idx] * (Jp[idx] - Jm[idx])
            );

        Jp[idx] += complex<double>(0.5, 0.0) * kr[idx] * F;
        Jm[idx] += complex<double>(-0.5, 0.0) * kr[idx] * F;
        Jz[idx] += complex<double>(0.0,-1.0) * kz[idx] * F;
    }
}

// 需要时可用编译选项覆盖
//#ifndef C2
//#define C2 8.9875517873681764e16      // c^2 (m^2/s^2)
//#endif
//#ifndef MU_0
//#define MU_0 1.2566370614359173e-6    // 真空磁导 (H/m)
//#endif

// ------------------------------------------------------------------
// 6. cuda_correct_currents_crossdeposition_standard
// ------------------------------------------------------------------
//此函数尚未验证正确性
extern "C" __global__
void cuda_correct_currents_crossdeposition_standard_kernel(
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    const complex<double>* __restrict__ rho_next_z,
    const complex<double>* __restrict__ rho_next_xy,
    complex<double>* __restrict__ Jp,
    complex<double>* __restrict__ Jm,
    complex<double>* __restrict__ Jz,
    const double* __restrict__ kz,
    const double* __restrict__ kr,
    const double inv_dt,
    const int Nz, const int Nr
){
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;

    complex<double> i_(0.0, 1.0);

    complex<double> Dz =
        i_ * kz[idx] * Jz[idx]
        + 0.5 * inv_dt * ( rho_next[idx] - rho_next_xy[idx]
                         + rho_next_z[idx] - rho_prev[idx] );

    complex<double> Dxy =
        kr[idx] * ( Jp[idx] - Jm[idx] )
        + 0.5 * inv_dt * ( rho_next[idx] - rho_next_z[idx]
                         + rho_next_xy[idx] - rho_prev[idx] );

    if (kr[idx] != 0.0) {
        double inv_kr = 1.0 / kr[idx];
        Jp[idx] += -0.5 * Dxy * inv_kr;
        Jm[idx] +=  0.5 * Dxy * inv_kr;
    }
    if (kz[idx] != 0.0) {
        double inv_kz = 1.0 / kz[idx];
        Jz[idx] += i_ * Dz * inv_kz;
    }
}

// ------------------------------------------------------------------
// 7. cuda_correct_currents_curlfree_comoving
// ------------------------------------------------------------------
extern "C" __global__
void cuda_correct_currents_curlfree_comoving_kernel(
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    complex<double>* __restrict__ Jp,
    complex<double>* __restrict__ Jm,
    complex<double>* __restrict__ Jz,
    const double* __restrict__ kz,
    const double* __restrict__ kr,
    const double* __restrict__ inv_k2,
    const complex<double>* __restrict__ j_corr_coef,
    const complex<double>* __restrict__ T_eb,
    const complex<double>* __restrict__ T_cc,
    const double inv_dt,   // 未在该核中直接使用，但保持签名一致
    const int Nz, const int Nr
){
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;

    complex<double> i_(0.0, 1.0);

    complex<double> F =
        -inv_k2[idx] * ( T_cc[idx] * j_corr_coef[idx]
                         * ( rho_next[idx] - rho_prev[idx] * T_eb[idx] )
                         + i_ * kz[idx] * Jz[idx]
                         + kr[idx] * ( Jp[idx] - Jm[idx] ) );

    Jp[idx] +=  complex<double>(0.5, 0.0)  * kr[idx] * F;
    Jm[idx] +=  complex<double>(-0.5, 0.0) * kr[idx] * F;
    Jz[idx] +=  complex<double>(0.0, -1.0) * kz[idx] * F;
}

extern "C" __global__
void cuda_correct_currents_curlfree_comoving_v0_kernel(
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    complex<double>* __restrict__ Jp,
    complex<double>* __restrict__ Jm,
    complex<double>* __restrict__ Jz,
    const double* __restrict__ kz,
    const double* __restrict__ kr,
    const double* __restrict__ inv_k2,
    const double* __restrict__ j_corr_coef,
    const complex<double>* __restrict__ T_eb,
    const complex<double>* __restrict__ T_cc,
    const double inv_dt,   // 未在该核中直接使用，但保持签名一致
    const int Nz, const int Nr
){
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;

    complex<double> i_(0.0, 1.0);

    complex<double> F =
        -inv_k2[idx] * ( T_cc[idx] * j_corr_coef[idx]
                         * ( rho_next[idx] - rho_prev[idx] * T_eb[idx] )
                         + i_ * kz[idx] * Jz[idx]
                         + kr[idx] * ( Jp[idx] - Jm[idx] ) );

    Jp[idx] +=  complex<double>(0.5, 0.0)  * kr[idx] * F;
    Jm[idx] +=  complex<double>(-0.5, 0.0) * kr[idx] * F;
    Jz[idx] +=  complex<double>(0.0, -1.0) * kz[idx] * F;
}

// ------------------------------------------------------------------
// 8. cuda_correct_currents_crossdeposition_comoving
// ------------------------------------------------------------------
extern "C" __global__
void cuda_correct_currents_crossdeposition_comoving_kernel(
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    const complex<double>* __restrict__ rho_next_z,
    const complex<double>* __restrict__ rho_next_xy,
    complex<double>* __restrict__ Jp,
    complex<double>* __restrict__ Jm,
    complex<double>* __restrict__ Jz,
    const double* __restrict__ kz,
    const double* __restrict__ kr,
    const complex<double>* __restrict__ j_corr_coef,
    const complex<double>* __restrict__ T_eb,
    const complex<double>* __restrict__ T_cc,
    const double inv_dt,   // 未直接使用，保留以匹配签名
    const int Nz, const int Nr
){
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;

    complex<double> i_(0.0, 1.0);

    complex<double> Dz =
        i_*kz[idx]*Jz[idx]
        + 0.5 * T_cc[idx]*j_corr_coef[idx] *
          ( rho_next[idx] - T_eb[idx]*rho_next_xy[idx]
            + rho_next_z[idx] - T_eb[idx]*rho_prev[idx] );

    complex<double> Dxy =
        kr[idx]*( Jp[idx] - Jm[idx] )
        + 0.5 * T_cc[idx]*j_corr_coef[idx] *
          ( rho_next[idx] + T_eb[idx]*rho_next_xy[idx]
            - rho_next_z[idx] - T_eb[idx]*rho_prev[idx] );

    if (kr[idx] != 0.0) {
        double inv_kr = 1.0 / kr[idx];
        Jp[idx] += -0.5 * Dxy * inv_kr;
        Jm[idx] +=  0.5 * Dxy * inv_kr;
    }
    if (kz[idx] != 0.0) {
        complex<double> inv_kz = 1.0 / kz[idx];
        Jz[idx] += i_ * Dz * inv_kz;
    }
}

extern "C" __global__
void cuda_correct_currents_crossdeposition_comoving_v0_kernel(
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    const complex<double>* __restrict__ rho_next_z,
    const complex<double>* __restrict__ rho_next_xy,
    complex<double>* __restrict__ Jp,
    complex<double>* __restrict__ Jm,
    complex<double>* __restrict__ Jz,
    const double* __restrict__ kz,
    const double* __restrict__ kr,
    const double* __restrict__ j_corr_coef,
    const complex<double>* __restrict__ T_eb,
    const complex<double>* __restrict__ T_cc,
    const double inv_dt,   // 未直接使用，保留以匹配签名
    const int Nz, const int Nr
){
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;

    complex<double> i_(0.0, 1.0);

    complex<double> Dz =
        i_*kz[idx]*Jz[idx]
        + 0.5 * T_cc[idx]*j_corr_coef[idx] *
          ( rho_next[idx] - T_eb[idx]*rho_next_xy[idx]
            + rho_next_z[idx] - T_eb[idx]*rho_prev[idx] );

    complex<double> Dxy =
        kr[idx]*( Jp[idx] - Jm[idx] )
        + 0.5 * T_cc[idx]*j_corr_coef[idx] *
          ( rho_next[idx] + T_eb[idx]*rho_next_xy[idx]
            - rho_next_z[idx] - T_eb[idx]*rho_prev[idx] );

    if (kr[idx] != 0.0) {
        double inv_kr = 1.0 / kr[idx];
        Jp[idx] += -0.5 * Dxy * inv_kr;
        Jm[idx] +=  0.5 * Dxy * inv_kr;
    }
    if (kz[idx] != 0.0) {
        complex<double> inv_kz = 1.0 / kz[idx];
        Jz[idx] += i_ * Dz * inv_kz;
    }
}

// ------------------------------------------------------------------
// 9. cuda_push_eb_standard
// ------------------------------------------------------------------
extern "C" __global__
void cuda_push_eb_standard_kernel(
    complex<double>* __restrict__ Ep,
    complex<double>* __restrict__ Em,
    complex<double>* __restrict__ Ez,
    complex<double>* __restrict__ Bp,
    complex<double>* __restrict__ Bm,
    complex<double>* __restrict__ Bz,
    const complex<double>* __restrict__ Jp,
    const complex<double>* __restrict__ Jm,
    const complex<double>* __restrict__ Jz,
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    const double* __restrict__ rho_prev_coef,
    const double* __restrict__ rho_next_coef,
    const double* __restrict__ j_coef,
    const double* __restrict__ C,
    const double* __restrict__ S_w,
    const double* __restrict__ kr,
    const double* __restrict__ kz,
    const double dt,
    bool use_true_rho,   // bool as int
    const int Nz, const int Nr,
    const double C2,
    const double epsilon_0,
    const double MU_0
){
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;

    complex<double> i_(0.0, 1.0);

    // Save E(t)
    complex<double> Ep_old = Ep[idx];
    complex<double> Em_old = Em[idx];
    complex<double> Ez_old = Ez[idx];

    // rho_diff
    complex<double> rho_diff;
    if (use_true_rho) {
        rho_diff = rho_next_coef[idx]*rho_next[idx]
                 - rho_prev_coef[idx]*rho_prev[idx];
    } else {
        complex<double> divE = kr[idx]*( Ep[idx] - Em[idx] )
                             + i_*kz[idx]*Ez[idx];
        complex<double> divJ = kr[idx]*( Jp[idx] - Jm[idx] )
                             + i_*kz[idx]*Jz[idx];
        rho_diff = (rho_next_coef[idx] - rho_prev_coef[idx]) * epsilon_0 * divE
                 - rho_next_coef[idx] * dt * divJ;
    }

    // Push E
    Ep[idx] = C[idx]*Ep[idx]
            + 0.5*kr[idx]*rho_diff
            + C2*S_w[idx] *
              ( -i_*complex<double>(0.5,0.0)*kr[idx]*Bz[idx]
                + kz[idx]*Bp[idx] - MU_0*Jp[idx] );

    Em[idx] = C[idx]*Em[idx]
            - 0.5*kr[idx]*rho_diff
            + C2 *S_w[idx] *
              ( -i_*0.5*kr[idx]*Bz[idx]
                - kz[idx]*Bm[idx] - MU_0*Jm[idx] );

    Ez[idx] = C[idx]*Ez[idx]
            - i_*kz[idx]*rho_diff
            + C2*S_w[idx] *
              ( i_*kr[idx]*Bp[idx] + i_*kr[idx]*Bm[idx] - MU_0*Jz[idx] );

    // Push B
    Bp[idx] = C[idx]*Bp[idx]
            - S_w[idx] * ( -i_*0.5*kr[idx]*Ez_old + kz[idx]*Ep_old )
            + j_coef[idx] * ( -i_*0.5*kr[idx]*Jz[idx] + kz[idx]*Jp[idx] );

    Bm[idx] = C[idx]*Bm[idx]
            - S_w[idx] * ( -i_*0.5*kr[idx]*Ez_old - kz[idx]*Em_old )
            + j_coef[idx] * ( -i_*0.5*kr[idx]*Jz[idx] - kz[idx]*Jm[idx] );

    Bz[idx] = C[idx]*Bz[idx]
            - S_w[idx] * (  i_*kr[idx]*Ep_old + i_*kr[idx]*Em_old )
            + j_coef[idx] * (  i_*kr[idx]*Jp[idx] + i_*kr[idx]*Jm[idx] );
}

// ------------------------------------------------------------------
// 10. cuda_push_eb_pml_standard
// ------------------------------------------------------------------
//尚未验证正确性
extern "C" __global__
void cuda_push_eb_pml_standard_kernel(
    complex<double>* __restrict__ Ep_pml,
    complex<double>* __restrict__ Em_pml,
    complex<double>* __restrict__ Bp_pml,
    complex<double>* __restrict__ Bm_pml,
    const complex<double>* __restrict__ Ez,
    const complex<double>* __restrict__ Bz,
    const double* __restrict__ C,
    const double* __restrict__ S_w,
    const double* __restrict__ kr,
    const double* __restrict__ kz,  // 未直接使用，保持签名一致
    const int Nz, const int Nr,
    const double C2
){
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;
    complex<double> i_(0.0, 1.0);
    // E (PML)
    Ep_pml[idx] = C[idx]*Ep_pml[idx]
                + C2*S_w[idx] * ( -i_*0.5*kr[idx]*Bz[idx] );

    Em_pml[idx] = C[idx]*Em_pml[idx]
                + C2*S_w[idx] * ( -i_*0.5*kr[idx]*Bz[idx] );

    // B (PML)
    Bp_pml[idx] = C[idx]*Bp_pml[idx]
                - S_w[idx] * ( -i_*0.5*kr[idx]*Ez[idx] );

    Bm_pml[idx] = C[idx]*Bm_pml[idx]
                - S_w[idx] * ( -i_*0.5*kr[idx]*Ez[idx] );
}
// 11
extern "C" __global__
void cuda_push_eb_comoving_kernel(
    complex<double>* __restrict__ Ep,
    complex<double>* __restrict__ Em,
    complex<double>* __restrict__ Ez,
    complex<double>* __restrict__ Bp,
    complex<double>* __restrict__ Bm,
    complex<double>* __restrict__ Bz,
    const complex<double>* __restrict__ Jp,
    const complex<double>* __restrict__ Jm,
    const complex<double>* __restrict__ Jz,
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    const complex<double>* __restrict__ rho_prev_coef,
    const complex<double>* __restrict__ rho_next_coef,
    const complex<double>* __restrict__ j_coef,
    const double* __restrict__ C,
    const double* __restrict__ S_w,
    const complex<double>* __restrict__ T_eb,
    const complex<double>* __restrict__ T_cc,
    const complex<double>* __restrict__ T_rho,
    const double* __restrict__ kr,
    const double* __restrict__ kz,
    const double dt,
    const double V,
    bool use_true_rho,
    const int Nz,
    const int Nr,
    const double C2,
    const double EPS0,
    const double MU0)
{
    const int iz = blockDim.x * blockIdx.x + threadIdx.x;
    const int ir = blockDim.y * blockIdx.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;

    const int idx = iz * Nr + ir;
    complex<double> i_(0.0, 1.0);
    // Save E fields (needed for B push)
    const complex<double> Ep_old = Ep[idx];
    const complex<double> Em_old = Em[idx];
    const complex<double> Ez_old = Ez[idx];

    // rho_diff
    complex<double> rho_diff;
    if (use_true_rho) {
        rho_diff = rho_next_coef[idx] * rho_next[idx]
                 - rho_prev_coef[idx] * rho_prev[idx];
    } else {
        // divE = kr*(Ep-Em) + i*kz*Ez
        complex<double> divE = kr[idx] * (Ep[idx] - Em[idx])
                             + i_ * kz[idx] * Ez[idx];
        // divJ = kr*(Jp-Jm) + i*kz*Jz
        complex<double> divJ = kr[idx] * (Jp[idx] - Jm[idx])
                             + i_ * kz[idx] * Jz[idx];
        rho_diff = ( T_eb[idx] * rho_next_coef[idx] - rho_prev_coef[idx] )
                   * EPS0 * divE
                 + T_rho[idx] * rho_next_coef[idx] * divJ;
    }

    // E push
    Ep[idx] = T_eb[idx]*C[idx]*Ep[idx]
            + 0.5*kr[idx] * rho_diff
            + j_coef[idx] * i_ * kz[idx] *V * Jp[idx]
            + C2*T_eb[idx]*S_w[idx]*( -i_*0.5*kr[idx] *Bz[idx]
                                               + kz[idx]*Bp[idx]
                                               - MU0*T_cc[idx]*Jp[idx] );

    Em[idx] = T_eb[idx]*C[idx]*Em[idx]
            - 0.5*kr[idx] * rho_diff
            + j_coef[idx] * i_* kz[idx]*V * Jm[idx]
            + C2*T_eb[idx]*S_w[idx]*( -i_ *0.5*kr[idx]*Bz[idx]
                                               - kz[idx]*Bm[idx]
                                               - MU0*T_cc[idx]*Jm[idx] );

    Ez[idx] = T_eb[idx]*C[idx]*Ez[idx]
            - i_* kz[idx] * rho_diff
            + j_coef[idx] * i_* kz[idx]*V * Jz[idx]
            + C2*T_eb[idx]*S_w[idx]* (i_ *  kr[idx]*Bp[idx]
                                               + i_* kr[idx]*Bm[idx]
                                               - MU0*T_cc[idx]*Jz[idx] );

    // B push (uses saved E)
    Bp[idx] = T_eb[idx]*C[idx]*Bp[idx]
            - T_eb[idx]*S_w[idx]*( -i_*0.5*kr[idx]*Ez_old
                      + kz[idx]*Ep_old )
            + j_coef[idx]*( -i_*0.5*kr[idx]*Jz[idx]
                  + kz[idx]*Jp[idx] );

    Bm[idx] = T_eb[idx]*C[idx]*Bm[idx]
            - T_eb[idx]*S_w[idx]*( -i_*0.5*kr[idx]*Ez_old
                      - kz[idx]*Em_old )
            + j_coef[idx]*(-i_*0.5*kr[idx]*Jz[idx]
                  - kz[idx]*Jm[idx] );

    Bz[idx] = T_eb[idx]*C[idx]*Bz[idx]
            - T_eb[idx]*S_w[idx]*(i_*kr[idx]*Ep_old
                      + i_* kr[idx]*Em_old )
            + j_coef[idx]*( i_* kr[idx]*Jp[idx]
                  + i_*kr[idx]*Jm[idx] );
}

extern "C" __global__
void cuda_push_eb_comoving_v0_kernel(
    complex<double>* __restrict__ Ep,
    complex<double>* __restrict__ Em,
    complex<double>* __restrict__ Ez,
    complex<double>* __restrict__ Bp,
    complex<double>* __restrict__ Bm,
    complex<double>* __restrict__ Bz,
    const complex<double>* __restrict__ Jp,
    const complex<double>* __restrict__ Jm,
    const complex<double>* __restrict__ Jz,
    const complex<double>* __restrict__ rho_prev,
    const complex<double>* __restrict__ rho_next,
    const double* __restrict__ rho_prev_coef,
    const double* __restrict__ rho_next_coef,
    const double* __restrict__ j_coef,
    const double* __restrict__ C,
    const double* __restrict__ S_w,
    const complex<double>* __restrict__ T_eb,
    const complex<double>* __restrict__ T_cc,
    const double* __restrict__ T_rho,
    const double* __restrict__ kr,
    const double* __restrict__ kz,
    const double dt,
    const double V,
    bool use_true_rho,
    const int Nz,
    const int Nr,
    const double C2,
    const double EPS0,
    const double MU0)
{
    const int iz = blockDim.x * blockIdx.x + threadIdx.x;
    const int ir = blockDim.y * blockIdx.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;

    const int idx = iz * Nr + ir;
    complex<double> i_(0.0, 1.0);
    // Save E fields (needed for B push)
    const complex<double> Ep_old = Ep[idx];
    const complex<double> Em_old = Em[idx];
    const complex<double> Ez_old = Ez[idx];

    // rho_diff
    complex<double> rho_diff;
    if (use_true_rho) {
        rho_diff = rho_next_coef[idx] * rho_next[idx]
                 - rho_prev_coef[idx] * rho_prev[idx];
    } else {
        // divE = kr*(Ep-Em) + i*kz*Ez
        complex<double> divE = kr[idx] * (Ep[idx] - Em[idx])
                             + i_ * kz[idx] * Ez[idx];
        // divJ = kr*(Jp-Jm) + i*kz*Jz
        complex<double> divJ = kr[idx] * (Jp[idx] - Jm[idx])
                             + i_ * kz[idx] * Jz[idx];
        rho_diff = ( T_eb[idx] * rho_next_coef[idx] - rho_prev_coef[idx] )
                   * EPS0 * divE
                 + T_rho[idx] * rho_next_coef[idx] * divJ;
    }

    // E push
    Ep[idx] = T_eb[idx]*C[idx]*Ep[idx]
            + 0.5*kr[idx] * rho_diff
            + j_coef[idx] * i_ * kz[idx] *V * Jp[idx]
            + C2*T_eb[idx]*S_w[idx]*( -i_*0.5*kr[idx] *Bz[idx]
                                               + kz[idx]*Bp[idx]
                                               - MU0*T_cc[idx]*Jp[idx] );

    Em[idx] = T_eb[idx]*C[idx]*Em[idx]
            - 0.5*kr[idx] * rho_diff
            + j_coef[idx] * i_* kz[idx]*V * Jm[idx]
            + C2*T_eb[idx]*S_w[idx]*( -i_ *0.5*kr[idx]*Bz[idx]
                                               - kz[idx]*Bm[idx]
                                               - MU0*T_cc[idx]*Jm[idx] );

    Ez[idx] = T_eb[idx]*C[idx]*Ez[idx]
            - i_* kz[idx] * rho_diff
            + j_coef[idx] * i_* kz[idx]*V * Jz[idx]
            + C2*T_eb[idx]*S_w[idx]* (i_ *  kr[idx]*Bp[idx]
                                               + i_* kr[idx]*Bm[idx]
                                               - MU0*T_cc[idx]*Jz[idx] );

    // B push (uses saved E)
    Bp[idx] = T_eb[idx]*C[idx]*Bp[idx]
            - T_eb[idx]*S_w[idx]*( -i_*0.5*kr[idx]*Ez_old
                      + kz[idx]*Ep_old )
            + j_coef[idx]*( -i_*0.5*kr[idx]*Jz[idx]
                  + kz[idx]*Jp[idx] );

    Bm[idx] = T_eb[idx]*C[idx]*Bm[idx]
            - T_eb[idx]*S_w[idx]*( -i_*0.5*kr[idx]*Ez_old
                      - kz[idx]*Em_old )
            + j_coef[idx]*(-i_*0.5*kr[idx]*Jz[idx]
                  - kz[idx]*Jm[idx] );

    Bz[idx] = T_eb[idx]*C[idx]*Bz[idx]
            - T_eb[idx]*S_w[idx]*(i_*kr[idx]*Ep_old
                      + i_* kr[idx]*Em_old )
            + j_coef[idx]*( i_* kr[idx]*Jp[idx]
                  + i_*kr[idx]*Jm[idx] );
}
// 12
//待验证正确性
extern "C" __global__
void cuda_push_eb_pml_comoving(
    complex<double>* __restrict__ Ep_pml,
    complex<double>* __restrict__ Em_pml,
    complex<double>* __restrict__ Bp_pml,
    complex<double>* __restrict__ Bm_pml,
    const complex<double>* __restrict__ Ez,
    const complex<double>* __restrict__ Bz,
    const double* __restrict__ C,
    const double* __restrict__ S_w,
    const complex<double>* __restrict__ T_eb,
    const double* __restrict__ kr,
    const double* __restrict__ kz,
    const int Nz,
    const int Nr,
    const double C2)
{
    const int iz = blockDim.x * blockIdx.x + threadIdx.x;
    const int ir = blockDim.y * blockIdx.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;
    complex<double> i_(0.0, 1.0);
    const complex<double> Teb = T_eb[idx];
    const double Cc  = C[idx];
    const double Sw  = S_w[idx];
    const double krv = kr[idx];

    // E push (split fields)
    Ep_pml[idx] = Teb*Cc*Ep_pml[idx]
                +C2*Teb*Sw*( -i_*0.5*krv*Bz[idx] );
    Em_pml[idx] = Teb*Cc*Em_pml[idx]
                + C2*Teb*Sw*( -i_*0.5*krv*Bz[idx] );

    // B push (split fields)
    Bp_pml[idx] = Teb*Cc*Bp_pml[idx]
                - Teb*Sw*( -i_*0.5*krv*Ez[idx] );
    Bm_pml[idx] = Teb*Cc*Bm_pml[idx]
                - Teb*Sw*( -i_*0.5*krv*Ez[idx] );
}
// 13
extern "C" __global__
void cuda_push_rho(
    complex<double>* __restrict__ rho_prev,
    complex<double>* __restrict__ rho_next,
    const int Nz,
    const int Nr)
{
    const int iz = blockDim.x * blockIdx.x + threadIdx.x;
    const int ir = blockDim.y * blockIdx.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    const int idx = iz * Nr + ir;
    rho_prev[idx] = rho_next[idx];
    rho_next[idx] = complex<double>(0.0, 0.0);
}
// 14
extern "C" __global__
void cuda_filter_scalar(
    complex<double>* __restrict__ field,
    const int Nz,
    const int Nr,
    const double* __restrict__ filter_array_z,
    const double* __restrict__ filter_array_r)
{
    int iz = blockDim.x * blockIdx.x + threadIdx.x;
    int ir = blockDim.y * blockIdx.y + threadIdx.y;
    if (iz >= Nz || ir >= Nr) return;
    int idx = iz * Nr + ir;
    field[idx] = filter_array_z[iz] * filter_array_r[ir] * field[idx];
}
//  15
extern "C" __global__
void cuda_filter_vector(
    complex<double>* fieldr,
    complex<double>* fieldt,
    complex<double>* fieldz,
    const int Nz,
    const int Nr,
    const double* filter_array_z,
    const double* filter_array_r)
{
    // 2D 线程索引
    int iz = blockDim.x * blockIdx.x + threadIdx.x;
    int ir = blockDim.y * blockIdx.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        double fz = filter_array_z[iz];
        double fr = filter_array_r[ir];
        double coeff = fz * fr;

        fieldr[idx] *= coeff;
        fieldt[idx] *= coeff;
        fieldz[idx] *= coeff;
    }
}