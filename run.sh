#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J mat_160
#SBATCH --time=72:00:00

module load intel/2022/mkl cuda/12.4 nccl/cuda-11.7/2.14.3 eigen/3.4 openmpi/4.0.5
cd build

#export OMP_NUM_THREADS=64
M=50034
geom=1
omega=1
leaf=128
padmis=1
rank=128
leveled_rank=10
acc=1e-8
admis=1
iters=10
max_iters=50
r1=0 #60
leveled_r1=0 #20
r2=0 #120
leveled_r2=0 #20

mpirun -n 32 ./main.app $M $geom $omega $leaf $padmis $rank $leveled_rank $acc $admis $iters $max_iters $r1 $leveled_r1 $r2 $leveled_r2
