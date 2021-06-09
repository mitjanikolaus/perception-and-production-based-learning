#!/bin/bash
#
#SBATCH --job-name=oracle
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/eval_oracle.out
#SBATCH --error=out/eval_oracle.out

source activate egg
python -u eval.py --batch_size 32 --max_len 25

