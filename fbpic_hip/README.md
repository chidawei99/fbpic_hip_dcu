# FBPIC with a HIP-compatible accelerator backend

This repository contains a modified version of the
[Fourier-Bessel Particle-In-Cell code (FBPIC)](https://github.com/fbpic/fbpic).
It adds a CuPy-based accelerator backend that can run the same FBPIC Python
workflow on NVIDIA CUDA GPUs and HIP-compatible accelerators, including Hygon
DCUs.

The high-level simulation interface, physical models, data layout, diagnostics,
and MPI longitudinal domain decomposition are retained from upstream FBPIC.
Performance-critical accelerator kernels are implemented as explicit C/C++
device kernels and launched through `cupy.RawKernel`/`cupy.RawModule`. The CuPy
build installed in the environment selects the CUDA or HIP runtime.

This source tree reports FBPIC version `0.26.1` in `fbpic/__init__.py`. It is a
research software snapshot associated with the manuscript:

> > A HIP-compatible accelerator backend for Fourier–Bessel particle-in-cell simulations on
> > CPU/DCU heterogeneous clusters

Mendeley Data DOI: **TBD**  
Article DOI: **TBD**

## Overview

FBPIC is a Particle-In-Cell (PIC) code for relativistic plasma physics. It is
particularly well suited to laser-wakefield acceleration and plasma-wakefield
acceleration problems with close-to-cylindrical symmetry.

FBPIC uses a spectral Fourier-Bessel decomposition of the fields in
quasi-cylindrical geometry. This combines much of the accuracy and stability of
spectral three-dimensional PIC methods with the reduced computational cost of a
cylindrical representation.

Major capabilities inherited from upstream FBPIC include:

- Moving windows
- Azimuthal-mode decomposition in cylindrical geometry
- Initial space-charge field calculation
- Mitigation of Numerical Cherenkov Radiation for relativistic bunches
- ADK field ionization
- Boosted-frame simulations and laboratory-frame diagnostics
- Galilean suppression of the Numerical Cherenkov Instability
- MPI parallelization by longitudinal domain decomposition
- openPMD-compatible HDF5 output

For the original algorithms and user documentation, see the
[FBPIC documentation](https://fbpic.github.io/) and the
[upstream repository](https://github.com/fbpic/fbpic).

## What is changed in this version

The accelerator execution layer has been changed while preserving the original
FBPIC user workflow:

- Performance-critical Numba-CUDA kernels were replaced by explicit C/C++
  kernels.
- The kernels are compiled and launched through CuPy's runtime-compilation
  interfaces.
- A common Python launch path is used for CUDA and HIP/DCU environments.
- Particle and field arrays remain resident on the accelerator during the PIC
  loop.
- Existing MPI decomposition and communication semantics are retained.
- External fields on an accelerator require an explicit device-function source
  string; see [External fields](#external-fields).

Except for the external-field interface described below, input scripts are used
in the same way as in upstream FBPIC. Existing examples are available under
`docs/source/example_input/`.

## CPU, CUDA, and HIP execution

The existing `Simulation(..., use_cuda=...)` argument is retained for
compatibility:

- `use_cuda=False` selects the CPU path. Numba is used to JIT-compile CPU
  routines.
- `use_cuda=True` selects the CuPy accelerator path. Depending on the installed
  CuPy package, kernels run through CUDA on an NVIDIA GPU or through HIP on a
  compatible accelerator/DCU.

The name `use_cuda` is therefore also used when running on a HIP/DCU platform.

With `use_cuda=True` and an available accelerator, simulation kernels do **not**
use Numba's CUDA or ROCm kernel backend. They are executed through CuPy and the
explicit C/C++ device kernels in this repository. Numba must nevertheless remain
installed because the package imports Numba, retains the CPU implementations,
and may read Numba configuration during initialization. In this accelerator
mode, Numba is an import/CPU-fallback dependency rather than the accelerator
kernel backend. The separate `numba-cuda` package required by some upstream
FBPIC CUDA configurations is not required by this modified accelerator backend.

## Installation

Do not run `pip install fbpic` for this software snapshot: that command installs
the upstream PyPI release rather than the modified backend in this archive.
Install from the directory that contains this README and `setup.py`.

The commands below consolidate the two installation notes supplied with this
archive. Package availability and module names can differ between clusters, so
adapt paths and scheduler commands to the local system.

### NVIDIA CUDA environment (CUDA 12.x / AutoDL example)

The recorded environment uses Python 3.10, NumPy 1.24, Numba 0.62.1, and the
CUDA 12.x build of CuPy.

```bash
conda create -n fbpic_cuda python=3.10 -y
conda activate fbpic_cuda

conda install -c conda-forge "numpy=1.24.*" pyfftw mpi4py -y
python -m pip install numba==0.62.1
python -m pip install cupy-cuda12x

conda install -c conda-forge mpi4py openmpi pyfftw --freeze-installed

cd /path/to/fbpic_base_mpi
python -m pip install -r requirements.txt
python -m pip install -e .

```

The installed CuPy package must match the CUDA runtime and GPU driver on the
target machine. Check the environment before running a simulation:

```bash
python -c "import fbpic, cupy as cp; print('FBPIC', fbpic.__version__); \
print('CuPy', cp.__version__); print('devices', cp.cuda.runtime.getDeviceCount())"
```

### HIP/DCU environment (DTK 25.04 example)

The recorded DCU environment uses Python 3.9, DTK 25.04, OpenMPI 4.1.5 , NumPy 1.24, and a vendor-provided CuPy 12.3.0 wheel. The wheel
name below is specific to that software stack; obtain the matching package from
the system administrator or accelerator software provider.

```bash
conda create -n fbpic_dcu python=3.9 -y
conda activate fbpic_dcu

module load compiler/dtk/25.04

conda install -c conda-forge "numpy=1.24.*" pyfftw --freeze-installed -y
python -m pip install numba

cd /path/to/package-directory
tar -xzf cupy-v12.3.0.dtk2504-27Feb2025.tar.gz
python -m pip install \
  ./cupy-v12.3.0.dtk2504-27Feb2025/cupy-12.3.0-cp39-cp39-linux_x86_64.whl

cd /path/to/fbpic_base_mpi
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install `mpi4py` against the MPI implementation used to launch the job. The
original DCU installation note used the following Conda command:

```bash
conda install -c conda-forge mpi4py openmpi pyfftw --freeze-installed
```

On a managed cluster, do not mix a site-provided OpenMPI module with a different
Conda OpenMPI runtime unless that combination is supported by the site. If the
loaded module must be used, follow the cluster's documented `mpi4py`
installation procedure.

Verify that the vendor CuPy package sees the DCU device:

```bash
python -c "import fbpic, cupy as cp; print('FBPIC', fbpic.__version__); \
print('CuPy', cp.__version__); print('devices', cp.cuda.runtime.getDeviceCount()); \
cp.show_config()"
```

### CPU-only environment

For a machine without an accelerator, install the CPU dependencies and set
`use_cuda=False` in the input script:

```bash
conda create -n fbpic_cpu python=3.10 -y
conda activate fbpic_cpu
conda install -c conda-forge "numpy=1.24.*" numba scipy h5py pyfftw mpi4py -y

cd /path/to/fbpic_base_mpi
python -m pip install -r requirements.txt
python -m pip install -e .
```

Intel CPU users may optionally install MKL if it is supported by their platform.

## Running simulations

Create or reuse a standard FBPIC input script and choose the execution path in
the `Simulation` constructor:

```python
from fbpic.main import Simulation

sim = Simulation(
    Nz, zmax, Nr, rmax, Nm, dt,
    zmin=zmin,
    use_cuda=True,  # CUDA GPU or HIP/DCU through the installed CuPy build
)
```

Run a single-process simulation with:

```bash
python fbpic_script.py
```

Run an MPI simulation with the launcher provided by the local MPI installation
or scheduler, for example:

```bash
mpirun -n 4 python fbpic_script.py
```

For multi-accelerator jobs, follow the cluster's documented rank-to-device
binding policy. FBPIC writes HDF5 diagnostics compatible with the
[openPMD standard](https://www.openpmd.org/), which can be inspected with
[openPMD-viewer](https://github.com/openPMD/openPMD-viewer).

## External fields

This is the one intentional user-facing difference from the upstream FBPIC
workflow. An `ExternalField` must receive both:

1. A Python function for the CPU path.
2. A raw device-function string for the CUDA/HIP accelerator path.

Use the syntax implemented in `tests/test_external_fields.py`. The Python
function receives seven arguments. The device function must be named
`field_func`, return a `double`, use the same first seven arguments in the same
order, and append `const double pi` and `const double c`:

```python
import math
from scipy.constants import c
from fbpic.lpa_utils.external_fields import ExternalField

amplitude = 1.0
length_scale = 1.0e-2
gamma_boost = None


def laser_func(F, x, y, z, t, amplitude, length_scale):
    """CPU expression for the external field."""
    return F + amplitude * math.cos(
        2 * math.pi * (z - c * t) / length_scale
    )


laser_func_code = r'''
__device__ __forceinline__
double field_func(
    double F,
    double x,
    double y,
    double z,
    double t,
    double amplitude,
    double length_scale,
    const double pi,
    const double c
)
{
    return F + amplitude * cos(2*pi*(z-c*t)/length_scale);
}
'''


sim.external_fields = [
    ExternalField(
        laser_func,
        laser_func_code,
        'Ex',
        amplitude,
        length_scale,
        gamma_boost=gamma_boost,
    )
]
```

Important details:

- Keep the device-function name exactly `field_func`; the generated wrapper
  calls that name.
- Keep the argument order and types exactly as shown.
- Use `return F + external_field` if the external field should be added to the
  field already gathered from the grid.
- The device string must use syntax accepted by the CUDA/HIP compiler used by
  the installed CuPy build.
- When `gamma_boost` is used, preserve the upstream requirement that the field
  expression be proportional to the `amplitude` argument.
- The complete executable example is `tests/test_external_fields.py`.



## Citation and attribution

If this modified backend contributes to a publication, cite both the associated
HIP/DCU backend paper and the original FBPIC algorithm paper:

1. J. Fan, R. He, H. Guo, and Q. Chen, "A HIP-compatible accelerator backend
   for Fourier-Bessel particle-in-cell simulations on DCU platforms."
   Article DOI: **TBD**.
2. R. Lehe, M. Kirchen, I. A. Andriyash, B. B. Godfrey, and J.-L. Vay,
   "A spectral, quasi-cylindrical and dispersion-free Particle-In-Cell
   algorithm," *Computer Physics Communications* 203 (2016) 66-82,
   [doi:10.1016/j.cpc.2016.02.007](https://doi.org/10.1016/j.cpc.2016.02.007).

The upstream FBPIC project was originally developed by Remi Lehe and Manuel
Kirchen and has received contributions from the FBPIC community. See the
[upstream README](https://github.com/fbpic/fbpic#research--attribution) for the
full contributor and algorithm-publication list.

## License

This is a modified derivative of FBPIC. Source files retain the upstream
copyright and license notices. Redistribution and use are governed by the
BSD-3-Clause-LBNL terms in `LICENSE.txt`. The names of the original institutions
and contributors may not be used to endorse this modified version without prior
written permission.

## Contact

For questions about this HIP/DCU backend and the associated manuscript, contact
Qiang Chen at `qiangchen@zzu.edu.cn`.

For questions about upstream FBPIC behavior, consult the
[official documentation](https://fbpic.github.io/) and the
[upstream issue tracker](https://github.com/fbpic/fbpic/issues).
