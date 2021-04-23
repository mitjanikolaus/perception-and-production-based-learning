#!/bin/bash
#
#SBATCH --job-name=caption
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_captioning_pre_train_for_interactive.out

source activate egg
python -u train_image_captioning.py --lr 0.0001 --n_epochs 30 --model interactive --log-frequency 700 #--fine-tune-resnet

# TODO: test with 30 epochs to make sure it's converged!

