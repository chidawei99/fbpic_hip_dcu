import os
import cupy as cp

# 获取当前文件路径，找到 .cu 文件
_kernel_path = os.path.join(os.path.dirname(__file__), "cupy_methods.cu")

with open(_kernel_path, 'r') as f:
    cupy_methods_code = f.read()

module = cp.RawModule(code=cupy_methods_code)

test_rng = module.get_function('test_rng')

get_photon_density_gaussian_cuda_c = module.get_function('get_photon_density_gaussian_cuda')
determine_scatterings_cuda_c = module.get_function('determine_scatterings_cuda')
scatter_photons_electrons_cuda_c = module.get_function('scatter_photons_electrons_cuda')