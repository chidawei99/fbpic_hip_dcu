
#define TWO_PI_FLOAT64 6.28318530717958647692


extern "C" {
__device__ __forceinline__
unsigned long long rotl(const unsigned long long x, const unsigned int k)
{
    return (x << k) | (x >> (64 - k));
}

__device__ __forceinline__
unsigned long long xoroshiro128p_next(
    unsigned long long* st_s0,
    unsigned long long* st_s1,
    const long long index
)
{
    unsigned long long s0 = st_s0[index];
    unsigned long long s1 = st_s1[index];
    const unsigned long long result = s0 + s1;

    s1 ^= s0;
    st_s0[index] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    st_s1[index] = rotl(s1, 36);

    return result;
}

__device__ __forceinline__
double uint64_to_unit_float64(const unsigned long long x)
{
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__
float uint64_to_unit_float32(const unsigned long long x)
{
    return (float)uint64_to_unit_float64(x);
}

__device__ __forceinline__
float xoroshiro128p_uniform_float32(
    unsigned long long* st_s0,
    unsigned long long* st_s1,
    const long long index
)
{
    return uint64_to_unit_float32(
        xoroshiro128p_next(st_s0, st_s1, index)
    );
}

__device__ __forceinline__
double xoroshiro128p_normal_float64(
    unsigned long long* st_s0,
    unsigned long long* st_s1,
    const long long index
)
{
    const float u1 = xoroshiro128p_uniform_float32(st_s0, st_s1, index);
    const float u2 = xoroshiro128p_uniform_float32(st_s0, st_s1, index);

    return sqrt(-2.0 * log((double)u1))
         * cos(TWO_PI_FLOAT64 * (double)u2);
}

__global__
void test_normal_float64(unsigned long long* st_s0, unsigned long long* st_s1, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = xoroshiro128p_normal_float64(st_s0, st_s1, i);
}
}

extern "C" {

// =====================
// device helpers
// =====================
__device__ inline void get_angles(
    double ux, double uy, double uz,
    double* theta_x, double* theta_y )
{
    *theta_x = atan2(ux, uz);
    *theta_y = atan2(uy, uz);
}

__device__ inline void get_linear_coefficients(
    double x, double xmin, double dx,
    int* ix, double* s0, double* s1 )
{
    double s_ix = (x - xmin) / dx;
    int i = (int)floor(s_ix);
    *s1 = s_ix - i;
    *s0 = 1.0 - (*s1);
    *ix = i;
}

__device__ void get_particle_radiation(
    double ux, double uy, double uz, double w,
    double Ex, double Ey, double Ez,
    double cBx, double cBy, double cBz,
    double gamma_inv,
    double Larmore_factor_density,
    double Larmore_factor_momentum,
    double SR_dxi,
    const double* SR_xi_data, int N_xi,
    const double* omega_ax, int N_omega,
    double* spect_loc,
    double* ux_ph, double* uy_ph, double* uz_ph, double E_MC )
{
    double gamma = 1.0 / gamma_inv;

    double beta_x = ux * gamma_inv;
    double beta_y = uy * gamma_inv;
    double beta_z = uz * gamma_inv;

    double dt_ux = -E_MC * (Ex + beta_y * cBz - beta_z * cBy);
    double dt_uy = -E_MC * (Ey + beta_z * cBx - beta_x * cBz);
    double dt_uz = -E_MC * (Ez + beta_x * cBy - beta_y * cBx);

    double dt_gamma = -E_MC * (Ex*beta_x + Ey*beta_y + Ez*beta_z);

    double dt_beta_x = (dt_ux - beta_x * dt_gamma) * gamma_inv;
    double dt_beta_y = (dt_uy - beta_y * dt_gamma) * gamma_inv;
    double dt_beta_z = (dt_uz - beta_z * dt_gamma) * gamma_inv;

    double Energy_norm =
        w * pow(gamma, 6) *
        ( dt_beta_x*dt_beta_x + dt_beta_y*dt_beta_y + dt_beta_z*dt_beta_z
        - pow(beta_y*dt_beta_z - beta_z*dt_beta_y, 2)
        - pow(beta_z*dt_beta_x - beta_x*dt_beta_z, 2)
        - pow(beta_x*dt_beta_y - beta_y*dt_beta_x, 2) );

    double Momentum_Larmor = Larmore_factor_momentum * Energy_norm / w;
    double u_abs_inv = 1.0 / sqrt(ux*ux + uy*uy + uz*uz);

    *ux_ph = Momentum_Larmor * ux * u_abs_inv;
    *uy_ph = Momentum_Larmor * uy * u_abs_inv;
    *uz_ph = Momentum_Larmor * uz * u_abs_inv;

    double dt_beta_abs2 =
        dt_beta_x*dt_beta_x + dt_beta_y*dt_beta_y + dt_beta_z*dt_beta_z;
    double beta_abs2_inv =
        1.0 / (beta_x*beta_x + beta_y*beta_y + beta_z*beta_z);

    double beta_dot =
        beta_x*dt_beta_x + beta_y*dt_beta_y + beta_z*dt_beta_z;

    double omega_c = 1.5 * pow(gamma, 3) * beta_abs2_inv *
        sqrt(dt_beta_abs2 - beta_dot*beta_dot * beta_abs2_inv);

    double d_omega = omega_ax[1] - omega_ax[0];

    if (omega_c < 4.0 * d_omega) {
        for (int i=0;i<N_omega;i++) spect_loc[i] = 0.0;
        *ux_ph = *uy_ph = *uz_ph = 0.0;
        return;
    }

    double omega_c_inv = 1.0 / omega_c;
    double Density_Larmore =
        Larmore_factor_density * Energy_norm * omega_c_inv;

    for (int i=0;i<N_omega;i++) {
        double xi = omega_ax[i] * omega_c_inv;
        double s = xi / SR_dxi;
        int ix = (int)floor(s);

        if (ix >= N_xi-1) {
            spect_loc[i] = 0.0;
        } else {
            double s1 = s - ix;
            double s0 = 1.0 - s1;
            spect_loc[i] =
                Density_Larmore *
                (SR_xi_data[ix]*s0 + SR_xi_data[ix+1]*s1);
        }
    }
}

// =====================
// kernel
// =====================
__global__ void gather_synchrotron_cuda(
    int N_batch, int batch_size, int Ntot,
    double* ux, double* uy, double* uz,
    double* Ex, double* Ey, double* Ez,
    double* Bx, double* By, double* Bz,
    double* w, double* gamma_inv,
    double Larmore_factor_density,
    double Larmore_factor_momentum,
    double gamma_cutoff_inv, bool radiation_reaction,
    const double* omega_ax, int N_omega,
    double SR_dxi, const double* SR_xi_data, int N_xi,
    double theta_x_min, double theta_x_max, double d_th_x,
    double theta_y_min, double theta_y_max, double d_th_y,
    double* spect_batch,
    unsigned long long* st_s0,
    unsigned long long* st_s1,
    double* radiation_data,
    int Ny,
    double C_LIGHT,
    double E_MC
    )
{
    int i_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_batch >= N_batch) return;

    double* spect_loc = spect_batch + i_batch * N_omega;

    int N_max = min((i_batch+1)*batch_size, Ntot);

    for (int ip=i_batch*batch_size; ip<N_max; ip++) {

        if (gamma_inv[ip] >= gamma_cutoff_inv) continue;

        double thx, thy;
        get_angles(ux[ip], uy[ip], uz[ip], &thx, &thy);

        double theta_diff = pow(2.0, -1.5) * gamma_inv[ip];
        thx += theta_diff * xoroshiro128p_normal_float64(
            st_s0, st_s1, i_batch);
        thy += theta_diff * xoroshiro128p_normal_float64(
            st_s0, st_s1, i_batch);

        if (thx <= theta_x_min || thx >= theta_x_max ||
            thy <= theta_y_min || thy >= theta_y_max)
            continue;

        int ix, iy;
        double s0x, s1x, s0y, s1y;
        get_linear_coefficients(thx, theta_x_min, d_th_x, &ix, &s0x, &s1x);
        get_linear_coefficients(thy, theta_y_min, d_th_y, &iy, &s0y, &s1y);

        double ux_ph, uy_ph, uz_ph;
        get_particle_radiation(
            ux[ip], uy[ip], uz[ip], w[ip],
            Ex[ip], Ey[ip], Ez[ip],
            C_LIGHT*Bx[ip], C_LIGHT*By[ip], C_LIGHT*Bz[ip],
            gamma_inv[ip],
            Larmore_factor_density,
            Larmore_factor_momentum,
            SR_dxi, SR_xi_data, N_xi,
            omega_ax, N_omega,
            spect_loc,
            &ux_ph, &uy_ph, &uz_ph, E_MC
        );

        if (radiation_reaction) {
            ux[ip] -= ux_ph;
            uy[ip] -= uy_ph;
            uz[ip] -= uz_ph;
            gamma_inv[ip] =
                1.0 / sqrt(1.0 + ux[ip]*ux[ip] +
                                     uy[ip]*uy[ip] +
                                     uz[ip]*uz[ip]);
        }

        for (int i=0;i<N_omega;i++) {
            double v = spect_loc[i];
            atomicAdd(&radiation_data[(ix  *Ny + iy  )*N_omega + i], v*s0x*s0y);
            atomicAdd(&radiation_data[((ix+1)*Ny + iy  )*N_omega + i], v*s1x*s0y);
            atomicAdd(&radiation_data[(ix  *Ny + iy+1)*N_omega + i], v*s0x*s1y);
            atomicAdd(&radiation_data[((ix+1)*Ny + iy+1)*N_omega + i], v*s1x*s1y);
        }
    }
}

} // extern "C"