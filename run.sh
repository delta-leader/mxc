#!/bin/bash
#YBATCH -r epyc-7502_1 
#SBATCH -N 1
#SBATCH -J ranks
#SBATCH --time=24:00:00

. /etc/profile.d/modules.sh
module load gcc/12.2
module load cmake/3.25.2
module load intel/2022/mkl
module load openmpi/4.0.5
module load cuda/12.2
module load eigen/3.4
module load nccl/cuda-11.7/2.14.3 

cd build
make -j

mpirun -n 8 ./main.app 262144 2 512 100 50 1e-8 h2 >> helmholtz_circle_262144_2.txt
