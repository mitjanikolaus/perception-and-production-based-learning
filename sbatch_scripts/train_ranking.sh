#!/bin/bash
#
#SBATCH --job-name=train-ranking
#
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_ranking_fine_tune_resnet_4.out

source activate egg
python -u train_image_sentence_ranking.py --lr 0.0001 --log-frequency 700 --n_epochs 30 --checkpoint-dir ~/data/visual_ref/checkpoints/ranking_fine_tune_resnet_4 --fine-tune-resnet

