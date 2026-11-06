import os
import cupy as cp

# 获取当前文件路径，找到 .cu 文件
_kernel_path = os.path.join(os.path.dirname(__file__), "cupy_methods.cu")

with open(_kernel_path, 'r', encoding='utf-8') as f:
    cupy_methods_code = f.read()

module = cp.RawModule(code=cupy_methods_code)

cuda_erase_scalar     = module.get_function('cuda_erase_scalar_kernel')

cuda_erase_vector     = module.get_function('cuda_erase_vector_kernel')

cuda_divide_scalar_by_volume    = module.get_function('cuda_divide_scalar_by_volume_kernel')

cuda_divide_vector_by_volume    = module.get_function('cuda_divide_vector_by_volume_kernel')

cuda_correct_currents_curlfree_standard = module.get_function('cuda_correct_currents_curlfree_standard_kernel')

cuda_correct_currents_crossdeposition_standard = module.get_function('cuda_correct_currents_crossdeposition_standard_kernel')

cuda_correct_currents_curlfree_comoving = module.get_function('cuda_correct_currents_curlfree_comoving_kernel')
cuda_correct_currents_curlfree_comoving_v0 = module.get_function('cuda_correct_currents_curlfree_comoving_v0_kernel')

cuda_correct_currents_crossdeposition_comoving = module.get_function('cuda_correct_currents_crossdeposition_comoving_kernel')
cuda_correct_currents_crossdeposition_comoving_v0 = module.get_function('cuda_correct_currents_crossdeposition_comoving_v0_kernel')

cuda_push_eb_standard = module.get_function('cuda_push_eb_standard_kernel')

cuda_push_eb_pml_standard = module.get_function('cuda_push_eb_pml_standard_kernel')

cuda_push_eb_comoving = module.get_function('cuda_push_eb_comoving_kernel')
cuda_push_eb_comoving_v0 = module.get_function('cuda_push_eb_comoving_v0_kernel')

cuda_push_eb_pml_comoving = module.get_function('cuda_push_eb_pml_comoving')

cuda_push_rho = module.get_function('cuda_push_rho')

cuda_filter_scalar_c = module.get_function('cuda_filter_scalar')

cuda_filter_vector_c = module.get_function('cuda_filter_vector')