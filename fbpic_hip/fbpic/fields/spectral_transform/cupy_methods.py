import os
import cupy as cp

# 获取当前文件路径，找到 .cu 文件
_kernel_path = os.path.join(os.path.dirname(__file__), "cupy_methods.cu")

with open(_kernel_path, 'r', encoding='utf-8') as f:
    cupy_methods_code = f.read()

module = cp.RawModule(code=cupy_methods_code)

cuda_copy_2dC_to_2dR = module.get_function( 'cuda_copy_2dC_to_2dR_kernel')
cuda_copy_2dR_to_2dC = module.get_function( 'cuda_copy_2dR_to_2dC_kernel')
cuda_copy_2d_to_1d   = module.get_function( 'cuda_copy_2d_to_1d_kernel')
cuda_copy_1d_to_2d   = module.get_function( 'cuda_copy_1d_to_2d_kernel')
cuda_rt_to_pm        = module.get_function( 'cuda_rt_to_pm_kernel')
cuda_pm_to_rt        = module.get_function( 'cuda_pm_to_rt_kernel')