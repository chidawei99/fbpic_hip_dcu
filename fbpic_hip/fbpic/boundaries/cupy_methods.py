import os
import cupy as cp

# 获取当前文件路径，找到 .cu 文件
_kernel_path = os.path.join(os.path.dirname(__file__), "cupy_methods.cu")

with open(_kernel_path, 'r', encoding='utf-8') as f:
    cupy_methods_code = f.read()

module = cp.RawModule(code=cupy_methods_code)

copy_vec_to_gpu_buffer       = module.get_function('copy_vec_to_gpu_buffer_kernel')

copy_pml_to_gpu_buffer       = module.get_function('copy_pml_to_gpu_buffer_kernel')

copy_scal_to_gpu_buffer      = module.get_function('copy_scal_to_gpu_buffer_kernel')

replace_vec_from_gpu_buffer  = module.get_function('replace_vec_from_gpu_buffer_kernel')

replace_pml_from_gpu_buffer  = module.get_function('replace_pml_from_gpu_buffer_kernel')

replace_scal_from_gpu_buffer = module.get_function('replace_scal_from_gpu_buffer_kernel')

add_vec_from_gpu_buffer      = module.get_function('add_vec_from_gpu_buffer_kernel')

add_scal_from_gpu_buffer     = module.get_function('add_scal_from_gpu_buffer_kernel')

cuda_damp_EB_left            = module.get_function('cuda_damp_EB_left_kernel')

cuda_damp_EB_left_pml        = module.get_function('cuda_damp_EB_left_pml_kernel')

cuda_damp_EB_right           = module.get_function('cuda_damp_EB_right_kernel')

cuda_damp_EB_right_pml           = module.get_function('cuda_damp_EB_right_pml_kernel')

