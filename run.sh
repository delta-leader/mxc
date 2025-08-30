#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J mat_160
#SBATCH --time=72:00:00

module load intel/2022/mkl cuda/12.4 nccl/cuda-11.7/2.14.3 eigen/3.4 openmpi/4.0.5
cd build

MAT=2530
export OMP_NUM_THREADS=64
for i in {1..50}; do
  mpirun -n 1 ./main.app ${MAT} 0 32 32 0 1e-12 fused2 $i > "../output/MAT_${MAT}/omega_${i}.txt"
done
