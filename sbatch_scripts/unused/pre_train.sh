#!/bin/bash
#
#SBATCH --job-name=pretrain
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/pre_train.out
#SBATCH --error=out/pre_train.out

source activate egg
python -u pre_train.py
