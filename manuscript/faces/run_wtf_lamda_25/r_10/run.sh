#!/bin/bash
#PBS -l walltime=0:30:00,select=1:ncpus=1:ngpus=1:mem=4gb
#PBS -N faces
#PBS -A st-schieb-1-gpu
#PBS -J 1-10
 
################################################################################

export NUMBA_CACHE_DIR=$PBS_O_WORKDIR

source ~/.bashrc
module load gcc/9.1.0
conda activate sdecouplings
cd $PBS_O_WORKDIR
SRAND=$PBS_ARRAY_INDEX
python faces.py --srcpath /scratch/st-schieb-1/zsteve/wtf/src --n_iter 25 --outfile "output_$SRAND" --r 10 --srand $SRAND --tol 1e-4 --split 5 --lamda 25
