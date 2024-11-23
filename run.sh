#!/bin/bash
#YBATCH -r epyc-7502_2
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
#module load nccl/cuda-11.7/2.14.3 

cd build
make -j

N=131072
admis=3

mpirun -n 16 ./main.app $N $admis 512 256 128 1e-10 h2 >> output/conv/helmholtz_circle_${N}_${admis}_h2.txt
mpirun -n 16 ./main.app $N $admis 512 256 128 1e-10 hss >> output/conv/helmholtz_circle_${N}_${admis}_hss.txt
