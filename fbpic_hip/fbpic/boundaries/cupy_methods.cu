#include <cupy/complex.cuh>
extern "C" __global__
void copy_vec_to_gpu_buffer_kernel(
    complex<double>* vec_buffer_l,
    complex<double>* vec_buffer_r,
    const complex<double>* grid_r,
    const complex<double>* grid_t,
    const complex<double>* grid_z,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    // thread indices
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;  // 要拷贝的 z 范围

    if (ir < Nr && iz < Nz_chunk) {
        // 左端拷贝
        if (copy_left) {
            int iz_left = nz_start + iz;
            vec_buffer_l[(3*m+0)*Nz_chunk*Nr + iz*Nr + ir] = grid_r[iz_left*Nr + ir];
            vec_buffer_l[(3*m+1)*Nz_chunk*Nr + iz*Nr + ir] = grid_t[iz_left*Nr + ir];
            vec_buffer_l[(3*m+2)*Nz_chunk*Nr + iz*Nr + ir] = grid_z[iz_left*Nr + ir];
        }
        // 右端拷贝
        if (copy_right) {
            int iz_right = Nz - nz_end + iz;
            vec_buffer_r[(3*m+0)*Nz_chunk*Nr + iz*Nr + ir] = grid_r[iz_right*Nr + ir];
            vec_buffer_r[(3*m+1)*Nz_chunk*Nr + iz*Nr + ir] = grid_t[iz_right*Nr + ir];
            vec_buffer_r[(3*m+2)*Nz_chunk*Nr + iz*Nr + ir] = grid_z[iz_right*Nr + ir];
        }
    }
}

extern "C" __global__
void copy_pml_to_gpu_buffer_kernel(
    complex<double>* vec_buffer_l,
    complex<double>* vec_buffer_r,
    const complex<double>* grid_r,
    const complex<double>* grid_t,
    const complex<double>* grid_z,
    const complex<double>* pml_r,
    const complex<double>* pml_t,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    // 线程索引
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;  // 要拷贝的 z 范围

    if (ir < Nr && iz < Nz_chunk) {
        // ---- 左端 ----
        if (copy_left) {
            int iz_left = nz_start + iz;
            vec_buffer_l[(5*m+0)*Nz_chunk*Nr + iz*Nr + ir] = grid_r[iz_left*Nr + ir];
            vec_buffer_l[(5*m+1)*Nz_chunk*Nr + iz*Nr + ir] = grid_t[iz_left*Nr + ir];
            vec_buffer_l[(5*m+2)*Nz_chunk*Nr + iz*Nr + ir] = grid_z[iz_left*Nr + ir];
            vec_buffer_l[(5*m+3)*Nz_chunk*Nr + iz*Nr + ir] = pml_r[iz_left*Nr + ir];
            vec_buffer_l[(5*m+4)*Nz_chunk*Nr + iz*Nr + ir] = pml_t[iz_left*Nr + ir];
        }

        // ---- 右端 ----
        if (copy_right) {
            int iz_right = Nz - nz_end + iz;
            vec_buffer_r[(5*m+0)*Nz_chunk*Nr + iz*Nr + ir] = grid_r[iz_right*Nr + ir];
            vec_buffer_r[(5*m+1)*Nz_chunk*Nr + iz*Nr + ir] = grid_t[iz_right*Nr + ir];
            vec_buffer_r[(5*m+2)*Nz_chunk*Nr + iz*Nr + ir] = grid_z[iz_right*Nr + ir];
            vec_buffer_r[(5*m+3)*Nz_chunk*Nr + iz*Nr + ir] = pml_r[iz_right*Nr + ir];
            vec_buffer_r[(5*m+4)*Nz_chunk*Nr + iz*Nr + ir] = pml_t[iz_right*Nr + ir];
        }
    }
}

extern "C" __global__
void copy_scal_to_gpu_buffer_kernel(
    complex<double>* scal_buffer_l,
    complex<double>* scal_buffer_r,
    const complex<double>* grid,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    // 线程索引
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;

    if (ir < Nr && iz < Nz_chunk) {
        // ---- 左端 ----
        if (copy_left) {
            int iz_left = nz_start + iz;
            scal_buffer_l[m*Nz_chunk*Nr + iz*Nr + ir] = grid[iz_left*Nr + ir];
        }

        // ---- 右端 ----
        if (copy_right) {
            int iz_right = Nz - nz_end + iz;
            scal_buffer_r[m*Nz_chunk*Nr + iz*Nr + ir] = grid[iz_right*Nr + ir];
        }
    }
}

extern "C" __global__
void replace_vec_from_gpu_buffer_kernel(
    const complex<double>* vec_buffer_l,
    const complex<double>* vec_buffer_r,
    complex<double>* grid_r,
    complex<double>* grid_t,
    complex<double>* grid_z,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;

    if (ir < Nr && iz < Nz_chunk) {
        // ---- 左端 ----
        if (copy_left) {
            int iz_left = iz;
            grid_r[iz_left*Nr + ir] = vec_buffer_l[(3*m+0)*Nz_chunk*Nr + iz*Nr + ir];
            grid_t[iz_left*Nr + ir] = vec_buffer_l[(3*m+1)*Nz_chunk*Nr + iz*Nr + ir];
            grid_z[iz_left*Nr + ir] = vec_buffer_l[(3*m+2)*Nz_chunk*Nr + iz*Nr + ir];
        }
        // ---- 右端 ----
        if (copy_right) {
            int iz_right = Nz - Nz_chunk + iz;
            grid_r[iz_right*Nr + ir] = vec_buffer_r[(3*m+0)*Nz_chunk*Nr + iz*Nr + ir];
            grid_t[iz_right*Nr + ir] = vec_buffer_r[(3*m+1)*Nz_chunk*Nr + iz*Nr + ir];
            grid_z[iz_right*Nr + ir] = vec_buffer_r[(3*m+2)*Nz_chunk*Nr + iz*Nr + ir];
        }
    }
}

extern "C" __global__
void replace_pml_from_gpu_buffer_kernel(
    const complex<double>* vec_buffer_l,
    const complex<double>* vec_buffer_r,
    complex<double>* grid_r,
    complex<double>* grid_t,
    complex<double>* grid_z,
    complex<double>* pml_r,
    complex<double>* pml_t,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;

    if (ir < Nr && iz < Nz_chunk) {
        // -------- 左端 --------
        if (copy_left) {
            int iz_left = iz;
            grid_r[iz_left*Nr + ir] = vec_buffer_l[(5*m+0)*Nz_chunk*Nr + iz*Nr + ir];
            grid_t[iz_left*Nr + ir] = vec_buffer_l[(5*m+1)*Nz_chunk*Nr + iz*Nr + ir];
            grid_z[iz_left*Nr + ir] = vec_buffer_l[(5*m+2)*Nz_chunk*Nr + iz*Nr + ir];
            pml_r [iz_left*Nr + ir] = vec_buffer_l[(5*m+3)*Nz_chunk*Nr + iz*Nr + ir];
            pml_t [iz_left*Nr + ir] = vec_buffer_l[(5*m+4)*Nz_chunk*Nr + iz*Nr + ir];
        }
        // -------- 右端 --------
        if (copy_right) {
            int iz_right = Nz - Nz_chunk + iz;
            grid_r[iz_right*Nr + ir] = vec_buffer_r[(5*m+0)*Nz_chunk*Nr + iz*Nr + ir];
            grid_t[iz_right*Nr + ir] = vec_buffer_r[(5*m+1)*Nz_chunk*Nr + iz*Nr + ir];
            grid_z[iz_right*Nr + ir] = vec_buffer_r[(5*m+2)*Nz_chunk*Nr + iz*Nr + ir];
            pml_r [iz_right*Nr + ir] = vec_buffer_r[(5*m+3)*Nz_chunk*Nr + iz*Nr + ir];
            pml_t [iz_right*Nr + ir] = vec_buffer_r[(5*m+4)*Nz_chunk*Nr + iz*Nr + ir];
        }
    }
}

extern "C" __global__
void replace_scal_from_gpu_buffer_kernel(
    const complex<double>* scal_buffer_l,
    const complex<double>* scal_buffer_r,
    complex<double>* grid,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;

    if (ir < Nr && iz < Nz_chunk) {
        // -------- 左端 --------
        if (copy_left) {
            int iz_left = iz;
            grid[iz_left*Nr + ir] =
                scal_buffer_l[m * Nz_chunk * Nr + iz * Nr + ir];
        }
        // -------- 右端 --------
        if (copy_right) {
            int iz_right = Nz - Nz_chunk + iz;
            grid[iz_right*Nr + ir] =
                scal_buffer_r[m * Nz_chunk * Nr + iz * Nr + ir];
        }
    }
}

extern "C" __global__
void add_vec_from_gpu_buffer_kernel(
    const complex<double>* vec_buffer_l,
    const complex<double>* vec_buffer_r,
    complex<double>* grid_r,
    complex<double>* grid_t,
    complex<double>* grid_z,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;

    if (ir < Nr && iz < Nz_chunk) {
        // ---------- 左端 ----------
        if (copy_left) {
            int iz_left = iz;
            grid_r[iz_left*Nr + ir] +=
                vec_buffer_l[(3*m+0)*Nz_chunk*Nr + iz*Nr + ir];
            grid_t[iz_left*Nr + ir] +=
                vec_buffer_l[(3*m+1)*Nz_chunk*Nr + iz*Nr + ir];
            grid_z[iz_left*Nr + ir] +=
                vec_buffer_l[(3*m+2)*Nz_chunk*Nr + iz*Nr + ir];
        }
        // ---------- 右端 ----------
        if (copy_right) {
            int iz_right = Nz - Nz_chunk + iz;
            grid_r[iz_right*Nr + ir] +=
                vec_buffer_r[(3*m+0)*Nz_chunk*Nr + iz*Nr + ir];
            grid_t[iz_right*Nr + ir] +=
                vec_buffer_r[(3*m+1)*Nz_chunk*Nr + iz*Nr + ir];
            grid_z[iz_right*Nr + ir] +=
                vec_buffer_r[(3*m+2)*Nz_chunk*Nr + iz*Nr + ir];
        }
    }
}

extern "C" __global__
void add_scal_from_gpu_buffer_kernel(
    const complex<double>* scal_buffer_l,
    const complex<double>* scal_buffer_r,
    complex<double>* grid,
    const int m,
    const bool copy_left,
    const bool copy_right,
    const int nz_start,
    const int nz_end,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    int Nz_chunk = nz_end - nz_start;

    if (ir < Nr && iz < Nz_chunk) {
        // --------- 左端 ----------
        if (copy_left) {
            int iz_left = iz;
            grid[iz_left*Nr + ir] +=
                scal_buffer_l[m*Nz_chunk*Nr + iz*Nr + ir];
        }
        // --------- 右端 ----------
        if (copy_right) {
            int iz_right = Nz - Nz_chunk + iz;
            grid[iz_right*Nr + ir] +=
                scal_buffer_r[m*Nz_chunk*Nr + iz*Nr + ir];
        }
    }
}

extern "C" __global__
void cuda_damp_EB_left_kernel(
    complex<double>* Er,
    complex<double>* Et,
    complex<double>* Ez,
    complex<double>* Br,
    complex<double>* Bt,
    complex<double>* Bz,
    const double* damp_array,
    const int nd,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (ir < Nr && iz < nd) {
        double damp_factor_left = damp_array[iz];

        int idx = iz * Nr + ir;

        Er[idx] *= damp_factor_left;
        Et[idx] *= damp_factor_left;
        Ez[idx] *= damp_factor_left;
        Br[idx] *= damp_factor_left;
        Bt[idx] *= damp_factor_left;
        Bz[idx] *= damp_factor_left;
    }
}

extern "C" __global__
void cuda_damp_EB_left_pml_kernel(
    complex<double>* Er_pml,
    complex<double>* Et_pml,
    complex<double>* Br_pml,
    complex<double>* Bt_pml,
    const double* damp_array,
    const int nd,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (ir < Nr && iz < nd) {
        double damp_factor_left = damp_array[iz];
        int idx = iz * Nr + ir;

        Er_pml[idx] *= damp_factor_left;
        Et_pml[idx] *= damp_factor_left;
        Br_pml[idx] *= damp_factor_left;
        Bt_pml[idx] *= damp_factor_left;
    }
}

extern "C" __global__
void cuda_damp_EB_right_kernel(
    complex<double>* Er,
    complex<double>* Et,
    complex<double>* Ez,
    complex<double>* Br,
    complex<double>* Bt,
    complex<double>* Bz,
    const double* damp_array,
    const int nd,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (ir < Nr && iz < nd) {
        double damp_factor_right = damp_array[iz];
        int iz_right = Nz - iz - 1;
        int idx = iz_right * Nr + ir;

        Er[idx] *= damp_factor_right;
        Et[idx] *= damp_factor_right;
        Ez[idx] *= damp_factor_right;
        Br[idx] *= damp_factor_right;
        Bt[idx] *= damp_factor_right;
        Bz[idx] *= damp_factor_right;
    }
}

extern "C" __global__
void cuda_damp_EB_right_pml_kernel(
    complex<double>* Er_pml,
    complex<double>* Et_pml,
    complex<double>* Br_pml,
    complex<double>* Bt_pml,
    const double* damp_array,
    const int nd,
    const int Nz,
    const int Nr
) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ir = blockIdx.y * blockDim.y + threadIdx.y;

    if (ir < Nr && iz < nd) {
        double damp_factor_right = damp_array[iz];
        int iz_right = Nz - iz - 1;
        int idx = iz_right * Nr + ir;

        Er_pml[idx] *= damp_factor_right;
        Et_pml[idx] *= damp_factor_right;
        Br_pml[idx] *= damp_factor_right;
        Bt_pml[idx] *= damp_factor_right;
    }
}



