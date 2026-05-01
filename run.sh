#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J mat_160
#SBATCH --time=72:00:00

module load intel/2022/mkl cuda/12.4 nccl/cuda-11.7/2.14.3 eigen/3.4 openmpi/4.0.5
cd build

#export OMP_NUM_THREADS=64
M=25
geom=3
omega=1
leaf=128
padmis=2
rank=128
leveled_rank=0
acc=1e-8
admis=2
iters=10
max_iters=20
r1=128 #60
#leveled_r1=0 #20
r2=128 #120
#leveled_r2=0 #20

mpirun -n 32 ./main.app $M $geom $omega $leaf $padmis $rank $leveled_rank $acc $admis $iters $max_iters $r1 $r2 &> ../output/salt/hidr/${M}_omega_${omega}_leaf_${leaf}_padmis_${padmis}_rank_${rank}_lrank_${leveled_rank}_admis_${admis}_r1_${r1}_r2_${r2}.txt
