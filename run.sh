#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J mat10114
#SBATCH --time=72:00:00

module load intel/2022/mkl cuda/12.4 nccl/cuda-11.7/2.14.3 eigen/3.4 openmpi/4.0.5
cd build

rank=120

mpirun -n 1 ./main.app 30338 0 200 ${rank} 0 1e-2 fused2 > "../output/mat10114_${rank}_fused2_hss.txt"
