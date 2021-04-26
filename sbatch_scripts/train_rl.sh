#!/bin/bash
#
#SBATCH --job-name=RL
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_rl.out

source activate egg
python -u train.py --lr 0.001 --max_len 20 --freeze-receiver --receiver-checkpoint ~/data/visual_ref/checkpoints/ranking_fine_tune_resnet/ranking.pt --print-sample-interactions --sender functional --log-frequency 100 --eval-frequency 1000 --weight-structural-loss 0 --out-checkpoints-dir out_rl # --eval-semantics

