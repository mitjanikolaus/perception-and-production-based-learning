#!/bin/bash
#
#SBATCH --job-name=train-ranking
#
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_ranking_transformer.out

source activate egg
python -u train_image_sentence_ranking.py --lr 0.0001 --n_epochs 15 --checkpoint-dir ~/data/visual_ref/checkpoints/ranking_transformer/ --language-encoder transformer

