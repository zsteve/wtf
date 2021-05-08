#!/bin/bash
#SBATCH --account=def-geof
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4000M               # memory (per node)
#SBATCH --time=0-00:30            # time (DD-HH:MM)
#SBATCH --array=1-10
source /home/syz/sdecouplings/bin/activate
SRAND=$RANDOM
echo $SRAND
nvidia-smi
python faces.py --srcpath /home/syz/syz/wtf/src --n_iter 25 --outfile "output_$SRAND" --r __R__ --srand $SRAND
