#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J salt_50k
#SBATCH --time=72:00:00

module load intel/2022/mkl cuda/12.4 nccl/cuda-11.7/2.14.3 eigen/3.4 openmpi/4.0.5
cd build

MAT=50
export OMP_NUM_THREADS=64
#for i in {1..50}; do
mpirun -n 8 ./main.app ${MAT} 3 1 512
#done
