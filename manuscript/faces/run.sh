#!/bin/bash
#SBATCH --account=def-geof
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8000M               # memory (per node)
#SBATCH --time=0-00:10            # time (DD-HH:MM)
source /home/syz/sdecouplings/bin/activate
python faces.py --srcpath /home/syz/syz/wtf/src --n_iter 10 --outfile output
