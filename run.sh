#!/bin/bash
#YBATCH -r a100_1
#SBATCH -N 1
#SBATCH -J experiments
#SBATCH --time=5:00:00

. /etc/profile.d/modules.sh
module load gcc/12.2
module load cmake/3.25.2
module load intel/2022/mkl
module load openmpi/4.0.5
module load cuda/12.2
module load eigen/3.4

cd build
make -j

mpirun -n 1 ./main.app 2048 1 128 100 1e-12 hss >> helmholtz_circle_2048_hss.txt
