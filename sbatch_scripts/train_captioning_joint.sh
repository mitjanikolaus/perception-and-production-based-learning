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
#SBATCH --output=out/train_captioning_joint_train_frac_0.01.out

source activate egg
python -u train_image_captioning.py --seed 7 --lr 0.0001 --log-frequency 700 --n-epochs 100 --model joint --fine-tune-resnet --training-set-size 0.01 --out-checkpoints-dir out_train_captioning_joint_train_frac_0.01 --eval-semantics

