#!/bin/bash
#
#SBATCH --job-name=train-lm
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_lm.out
#SBATCH --error=out/train_lm.out

source activate egg
python -u train_language_model.py --lr 0.0001 --n_epochs 50

