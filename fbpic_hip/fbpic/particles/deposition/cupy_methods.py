import os
import cupy as cp

# 获取当前文件路径，找到 .cu 文件
_kernel_path = os.path.join(os.path.dirname(__file__), "cupy_methods.cu")

with open(_kernel_path, 'r') as f:
    cupy_methods_code = f.read()

module = cp.RawModule(code=cupy_methods_code)

deposit_rho_gpu_linear = module.get_function('deposit_rho_gpu_linear')

deposit_J_gpu_linear_c = module.get_function('deposit_J_gpu_linear')

deposit_rho_gpu_cubic = module.get_function('deposit_rho_gpu_cubic')

deposit_J_gpu_cubic_c = module.get_function('deposit_J_gpu_cubic')