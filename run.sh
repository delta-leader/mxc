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
#module load nccl/cuda-11.7/2.14.3 

cd build
make -j

N=65536
admis=3
geom=sphere
rank=64
rplus=32

mpirun -n 1 ./main.app $N $admis 512 $rank $rplus 1e-10 h2 $geom >> output/conv/helmholtz_${geom}_${N}_${admis}_${rank}_${rplus}_h2.txt
mpirun -n 1 ./main.app $N $admis 512 $rank $rplus 1e-10 hss $geom >> output/conv/helmholtz_${geom}_${N}_${admis}_${rank}_${rplus}_hss.txt
