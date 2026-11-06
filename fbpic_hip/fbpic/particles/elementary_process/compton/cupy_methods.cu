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
double xoroshiro128p_uniform_float64(
    unsigned long long* st_s0,
    unsigned long long* st_s1,
    const long long index
)
{
    return uint64_to_unit_float64(
        xoroshiro128p_next(st_s0, st_s1, index)
    );
}

__global__
void test_rng(unsigned long long* st_s0, unsigned long long* st_s1, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = xoroshiro128p_uniform_float64(st_s0, st_s1, i);
}
}

extern "C" __global__
void get_photon_density_gaussian_cuda(
    double* photon_n,
    const int elec_Ntot,
    const double* elec_x,
    const double* elec_y,
    const double* elec_z,
    const double ct,
    const double photon_n_lab_max,
    const double inv_laser_waist2,
    const double inv_laser_ctau2,
    const double laser_initial_z0,
    const double gamma_boost,
    const double beta_boost
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elec_Ntot){

        // Transform to lab frame
        double elec_zlab  = gamma_boost * ( elec_z[i] + beta_boost * ct );
        double elec_ctlab = gamma_boost * ( ct + beta_boost * elec_z[i] );

        // Photon density in lab frame
        double dx2 = elec_x[i]*elec_x[i] + elec_y[i]*elec_y[i];
        double dz  = elec_zlab - laser_initial_z0 + elec_ctlab;

        double photon_n_lab = photon_n_lab_max * exp(
            -2.0 * inv_laser_waist2 * dx2
            -2.0 * inv_laser_ctau2   * dz * dz
        );

        // Transform back to simulation frame
        photon_n[i] = gamma_boost * photon_n_lab * (1.0 + beta_boost);
    }
}

__device__ __forceinline__
double get_scattering_probability_device(
    const double dt,
    const double elec_ux,
    const double elec_uy,
    const double elec_uz,
    const double elec_inv_gamma,
    const double photon_n,
    const double photon_p,
    const double photon_beta_x,
    const double photon_beta_y,
    const double photon_beta_z,
    const double c,
    const double INV_MC,
    const double PI_RE_2
){
    // Electron gamma
    double gamma = 1.0 / elec_inv_gamma;

    // Lorentz transform of photon flux
    double transform_factor =
          gamma
        - elec_ux * photon_beta_x
        - elec_uy * photon_beta_y
        - elec_uz * photon_beta_z;

    double photon_n_rest = photon_n * transform_factor;
    double photon_p_rest = photon_p * transform_factor;

    // Klein–Nishina cross section
    double k = photon_p_rest * INV_MC;

    double f1 =
        2.0 * (2.0 + k * (1.0 + k) * (8.0 + k))
        / (k * k * (1.0 + 2.0 * k) * (1.0 + 2.0 * k));

    double f2 =
        (2.0 + k * (2.0 - k))
        * log(1.0 + 2.0 * k)
        / (k * k * k);

    double sigma = PI_RE_2 * (f1 - f2);

    // Proper time
    double proper_dt_rest = dt * elec_inv_gamma;

    // Scattering probability
    return 1.0 - exp(-sigma * photon_n_rest * c * proper_dt_rest);
}

// =================  kernel =================
extern "C" __global__
void determine_scatterings_cuda(
    const int N_batch,
    const int batch_size,
    const int elec_Ntot,
    long long* nscatter_per_elec,
    long long* nscatter_per_batch,
    unsigned long long* st_s0,
    unsigned long long* st_s1,
    const double dt,
    const double* elec_ux,
    const double* elec_uy,
    const double* elec_uz,
    const double* elec_inv_gamma,
    const int ratio_w_electron_photon,
    const double* photon_n,
    const double photon_p,
    const double photon_beta_x,
    const double photon_beta_y,
    const double photon_beta_z,
    const double c,
    const double INV_MC,
    const double PI_RE_2
){
    int i_batch = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_batch >= N_batch) return;

    nscatter_per_batch[i_batch] = 0;

    int ip = i_batch * batch_size;
    int N_max = (i_batch + 1) * batch_size;
    if (N_max > elec_Ntot) N_max = elec_Ntot;

    while (ip < N_max){

        nscatter_per_elec[ip] = 0;

        double p = get_scattering_probability_device(
            dt,
            elec_ux[ip],
            elec_uy[ip],
            elec_uz[ip],
            elec_inv_gamma[ip],
            photon_n[ip],
            photon_p,
            photon_beta_x,
            photon_beta_y,
            photon_beta_z,
            c,
            INV_MC,
            PI_RE_2
        );

        double r = xoroshiro128p_uniform_float64(st_s0, st_s1, i_batch);
        int nscatter = (int)(p * ratio_w_electron_photon + r);

        nscatter_per_elec[ip] = nscatter;
        nscatter_per_batch[i_batch] += nscatter;

        ip += 1;
    }
}

// ================= Lorentz transform =================
__device__ __forceinline__
void lorentz_transform(
    const double p_in,
    const double px_in,
    const double py_in,
    const double pz_in,
    const double gamma,
    const double beta,
    const double nx,
    const double ny,
    const double nz,
    double* p_out,
    double* px_out,
    double* py_out,
    double* pz_out
){
    double p_parallel_in =
        nx * px_in + ny * py_in + nz * pz_in;

    *p_out = gamma * (p_in - beta * p_parallel_in);
    double p_parallel_out =
        gamma * (p_parallel_in - beta * p_in);

    *px_out = px_in + nx * (p_parallel_out - p_parallel_in);
    *py_out = py_in + ny * (p_parallel_out - p_parallel_in);
    *pz_out = pz_in + nz * (p_parallel_out - p_parallel_in);
}

// =================  kernel =================
extern "C" __global__
void scatter_photons_electrons_cuda(
    const int N_batch,
    const int batch_size,
    const int photon_old_Ntot,
    const int elec_Ntot,
    const long long* cumul_nscatter_per_batch,
    const long long* nscatter_per_elec,
    unsigned long long* st_s0,
    unsigned long long* st_s1,
    const double photon_p,
    const double photon_px,
    const double photon_py,
    const double photon_pz,
    double* photon_x,
    double* photon_y,
    double* photon_z,
    double* photon_inv_gamma,
    double* photon_ux,
    double* photon_uy,
    double* photon_uz,
    double* photon_w,
    const double* elec_x,
    const double* elec_y,
    const double* elec_z,
    const double* elec_inv_gamma,
    double* elec_ux,
    double* elec_uy,
    double* elec_uz,
    const double* elec_w,
    const double inv_ratio_w_elec_photon,
    const double PI,
    const double INV_MC
){
    int i_batch = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_batch >= N_batch) return;

    long long i_photon =
        photon_old_Ntot + cumul_nscatter_per_batch[i_batch];

    int N_max = (i_batch + 1) * batch_size;
    if (N_max > elec_Ntot) N_max = elec_Ntot;

    for (int i_elec = i_batch * batch_size;
         i_elec < N_max;
         i_elec++){

        if (nscatter_per_elec[i_elec] > 0){

            // ===== Electron properties =====
            double inv_gamma = elec_inv_gamma[i_elec];
            double gamma = 1.0 / inv_gamma;

            double ux = elec_ux[i_elec];
            double uy = elec_uy[i_elec];
            double uz = elec_uz[i_elec];

            double u = sqrt(ux*ux + uy*uy + uz*uz);
            double beta = u * inv_gamma;

            double nx, ny, nz;
            if (u != 0.0){
                double inv_u = 1.0 / u;
                nx = ux * inv_u;
                ny = uy * inv_u;
                nz = uz * inv_u;
            } else {
                nx = 0.0; ny = 0.0; nz = 1.0;
            }

            // ===== Incoming photon to rest frame =====
            double photon_rest_p;
            double photon_rest_px;
            double photon_rest_py;
            double photon_rest_pz;

            lorentz_transform(
                photon_p, photon_px, photon_py, photon_pz,
                gamma, beta, nx, ny, nz,
                &photon_rest_p,
                &photon_rest_px,
                &photon_rest_py,
                &photon_rest_pz
            );

            double cos_theta = photon_rest_pz / photon_rest_p;
            double sin_theta, cos_phi, sin_phi;

            if (cos_theta*cos_theta < 1.0){
                sin_theta = sqrt(1.0 - cos_theta*cos_theta);
                double inv_pxy = 1.0 / (sin_theta * photon_rest_p);
                cos_phi = photon_rest_px * inv_pxy;
                sin_phi = photon_rest_py * inv_pxy;
            } else {
                sin_theta = 0.0;
                cos_phi = 1.0;
                sin_phi = 0.0;
            }

            double new_photon_p;
            double new_photon_px;
            double new_photon_py;
            double new_photon_pz;
            // ===== Loop over scatterings =====
            for (int i_scat = 0;
                 i_scat < nscatter_per_elec[i_elec];
                 i_scat++){

                double k = photon_rest_p * INV_MC;
                double c0 =
                    2.0 * (2.0*k*k + 2.0*k + 1.0)
                    / pow(2.0*k + 1.0, 3.0);
                double b = (2.0 + c0) / (2.0 - c0);
                double a = 2.0*b - 1.0;

                double x;
                while (1){
                    double r1 =
                        xoroshiro128p_uniform_float64(st_s0, st_s1, i_batch);
                    x = b - (b + 1.0) * pow(0.5*c0, r1);

                    double h = a / (b - x);
                    double factor = 1.0 + k*(1.0 - x);
                    double f =
                        ((1.0 + x*x)*factor + k*k*(1.0 - x)*(1.0 - x))
                        / (factor*factor*factor);

                    double r2 =
                        xoroshiro128p_uniform_float64(st_s0, st_s1, i_batch);
                    if (r2 < f/h) break;
                }

                double new_photon_rest_p =
                    photon_rest_p / (1.0 + k*(1.0 - x));

                double cos_theta_s = x;
                double sin_theta_s = sqrt(1.0 - x*x);

                double r3 =
                    xoroshiro128p_uniform_float64(st_s0, st_s1, i_batch);
                double phi_s = 2.0 * PI * r3;

                double cos_phi_s = cos(phi_s);
                double sin_phi_s = sin(phi_s);

                double pX =
                    new_photon_rest_p * sin_theta_s * cos_phi_s;
                double pY =
                    new_photon_rest_p * sin_theta_s * sin_phi_s;
                double pZ =
                    new_photon_rest_p * cos_theta_s;

                double new_photon_rest_px =
                    sin_theta*cos_phi*pZ
                  + cos_theta*cos_phi*pX
                  - sin_phi*pY;

                double new_photon_rest_py =
                    sin_theta*sin_phi*pZ
                  + cos_theta*sin_phi*pX
                  + cos_phi*pY;

                double new_photon_rest_pz =
                    cos_theta*pZ - sin_theta*pX;

                lorentz_transform(
                    new_photon_rest_p,
                    new_photon_rest_px,
                    new_photon_rest_py,
                    new_photon_rest_pz,
                    gamma, beta, -nx, -ny, -nz,
                    &new_photon_p,
                    &new_photon_px,
                    &new_photon_py,
                    &new_photon_pz
                );

                photon_x[i_photon] = elec_x[i_elec];
                photon_y[i_photon] = elec_y[i_elec];
                photon_z[i_photon] = elec_z[i_elec];
                photon_w[i_photon] =
                    elec_w[i_elec] * inv_ratio_w_elec_photon;

                photon_ux[i_photon] = new_photon_px;
                photon_uy[i_photon] = new_photon_py;
                photon_uz[i_photon] = new_photon_pz;
                photon_inv_gamma[i_photon] = 1.0 / new_photon_p;

                i_photon += 1;
            }

            // ===== Electron recoil =====
            double r =
                xoroshiro128p_uniform_float64(st_s0, st_s1, i_batch);
            if (r < inv_ratio_w_elec_photon){
                elec_ux[i_elec] += INV_MC * (photon_px - new_photon_px);
                elec_uy[i_elec] += INV_MC * (photon_py - new_photon_py);
                elec_uz[i_elec] += INV_MC * (photon_pz - new_photon_pz);
            }
        }
    }
}

