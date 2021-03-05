#!/bin/bash
#
#SBATCH --job-name=train-SAT
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_captioning_sat.out
#SBATCH --error=out/train_captioning_sat.out

source activate egg
python -u train_image_captioning.py --lr 0.0001 --n_epochs 15 --model show_attend_and_tell #--fine-tune-resnet

