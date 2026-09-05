#!/bin/sh
#$ -cwd
#$ -l node_f=3
#$ -l h_rt=12:00:00
#$ -N solve_sphere

module load cmake/3.28.3
module load intel/2025.0.0
module load cuda/12.8.0
module load gcc/14.2.0
module load openmpi/5.0.7-gcc
module load nccl/2.26.2

cd build

hostfile="hostfile_${RANDOM}.tmp"
cut -d " " -f1 $PE_HOSTFILE >> $hostfile
cat $hostfile

M=169798
geom=1
omega=7.115
leaf=128
padmis=2
rank=128
leveled_rank=0
acc=1e-8
admis=2
iters=10
max_iters=20
s1=128
s2=128
write_result=0
read_folder=""
runs=5

cp=12
p=48

export OMP_NUM_THREADS=${cp}
echo "f_node=4"
echo "${p} processes, ${cp} cores per process"

mpirun -n $p -x LD_LIBRARY_PATH --hostfile $hostfile --map-by numa:PE=$cp --report-bindings ./main.app $M $geom $omega $leaf $padmis $rank $leveled_rank $acc $admis $iters $max_iters $s1 $s2 $write_result $read_folder $runs &> ../output_salt.txt


rm $hostfile
