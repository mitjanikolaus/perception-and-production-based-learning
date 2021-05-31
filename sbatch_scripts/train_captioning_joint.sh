#!/bin/bash
#
#SBATCH --job-name=train-joint
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/train_captioning_joint.out

source activate egg
python -u train_image_captioning.py --lr 0.0001 --log-frequency 700 --n-epochs 25 --model joint --fine-tune-resnet --eval-semantics

