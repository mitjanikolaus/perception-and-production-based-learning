#!/bin/bash
#
#SBATCH --job-name=xsl
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
###SBATCHCOMMENT --constraint=cuda75|cuda61
#SBATCH --output=out/train_st.out

source activate egg
python -u train_image_captioning.py --seed 4 --lr 0.0001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --out-checkpoints-dir results_1/xsl/4/ --training-set-size 0.01 --fine-tune-resnet --eval-semantics --frequency-rl-updates 0

