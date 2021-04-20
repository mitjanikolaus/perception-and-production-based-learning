#!/bin/bash
#
#SBATCH --job-name=train-joint
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_joint.out

source activate egg
python -u train_image_captioning.py --lr 0.0001 --n_epochs 30 --model joint --log-frequency 500 #--fine-tune-resnet

