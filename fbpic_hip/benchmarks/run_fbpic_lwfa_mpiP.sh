#!/bin/bash
#SBATCH -J fbpic_lwfa
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -c 16
#SBATCH --gres=dcu:4
#SBATCH -o fbpic_lwfa_%j.out
#SBATCH -e fbpic_lwfa_%j.err

set -e

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Number of nodes: ${SLURM_JOB_NUM_NODES}"
echo "Number of tasks: ${SLURM_NTASKS}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"

echo "Allocated nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"

# 避免每个 MPI rank 内部再开过多线程


# 初始化 conda
# 如果你们集群 conda 初始化路径不同，可以把这一行改成 source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

conda activate numba_dtk_openmpi

module load compiler/dtk/25.04


cd /public/home/acvlei0ijb/project/fbpic_base_mpi/tests

echo "Current dir: $(pwd)"
echo "Python: $(which python)"
echo "mpirun: $(which mpirun)"
python -V
mpirun --version | head -n 3

echo "Start FBPIC run at: $(date)"


export MPIP=" -f ."
mpirun -np ${SLURM_NTASKS} \
    -x MPIP \
    -x MPI4PY_RC_THREADS=0 \
    -x LD_PRELOAD=/public/home/acvlei0ijb/project/mpiP/libmpiP.so \
    python lwfa_script2.py


echo "End FBPIC run at: $(date)"