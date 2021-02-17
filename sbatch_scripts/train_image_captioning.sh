#!/bin/bash
#
#SBATCH --job-name=train-IC
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_image_captioning.out
#SBATCH --error=out/train_image_captioning.out

source activate egg
python -u train_image_captioning.py --lr 0.00001 --n_epochs 50 --fine-tune-resnet

