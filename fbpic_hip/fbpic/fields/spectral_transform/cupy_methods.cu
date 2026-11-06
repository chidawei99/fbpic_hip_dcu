#include <cupy/complex.cuh>

extern "C" __global__
void cuda_copy_2dC_to_2dR_kernel(
    const complex<double>* __restrict__ array_in,
    double* __restrict__ array_out,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx_in = iz * Nr + ir;
        int idx_out_real = iz * Nr + ir;
        int idx_out_imag = (iz + Nz) * Nr + ir;

        complex<double> v = array_in[idx_in];
        array_out[idx_out_real] = v.real();
        array_out[idx_out_imag] = v.imag();
    }
}

// 2. 从 2D 实数 -> 2D 复数
extern "C" __global__
void cuda_copy_2dR_to_2dC_kernel(
    const double* __restrict__ array_in,
    complex<double>* __restrict__ array_out,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx_out = iz * Nr + ir;
        int idx_in_real = iz * Nr + ir;
        int idx_in_imag = (iz + Nz) * Nr + ir;

        double real_val = array_in[idx_in_real];
        double imag_val = array_in[idx_in_imag];
        array_out[idx_out] = complex<double>(real_val, imag_val);
    }
}

// 3. 从 2D 复数 -> 1D 复数
extern "C" __global__
void cuda_copy_2d_to_1d_kernel(
    const complex<double>* __restrict__ array_2d,
    complex<double>* __restrict__ array_1d,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx_in = iz * Nr + ir;
        int idx_out = iz + Nz * ir;  // 保持 NumPy 的存储顺序
        array_1d[idx_out] = array_2d[idx_in];
    }
}

// 4. 从 1D 复数 -> 2D 复数
extern "C" __global__
void cuda_copy_1d_to_2d_kernel(
    const complex<double>* __restrict__ array_1d,
    complex<double>* __restrict__ array_2d,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx_out = iz * Nr + ir;
        int idx_in = iz + Nz * ir;
        array_2d[idx_out] = array_1d[idx_in];
    }
}

// (1) r,t -> p,m
extern "C" __global__
void cuda_rt_to_pm_kernel(
    const complex<double>* __restrict__ buffer_r,
    const complex<double>* __restrict__ buffer_t,
    complex<double>* __restrict__ buffer_p,
    complex<double>* __restrict__ buffer_m,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        complex<double> value_r = buffer_r[idx];
        complex<double> value_t = buffer_t[idx];

        buffer_p[idx] = complex<double>(0.5, 0.0) * (value_r - complex<double>(0.0,1.0) * value_t);
        buffer_m[idx] = complex<double>(0.5, 0.0) * (value_r + complex<double>(0.0,1.0) * value_t);
    }
}

// (2) p,m -> r,t
extern "C" __global__
void cuda_pm_to_rt_kernel(
    const complex<double>* __restrict__ buffer_p,
    const complex<double>* __restrict__ buffer_m,
    complex<double>* __restrict__ buffer_r,
    complex<double>* __restrict__ buffer_t,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz < Nz && ir < Nr) {
        int idx = iz * Nr + ir;
        complex<double> value_p = buffer_p[idx];
        complex<double> value_m = buffer_m[idx];

        buffer_r[idx] = value_p + value_m;
        buffer_t[idx] = complex<double>(0.0,1.0) * (value_p - value_m);
    }
}

