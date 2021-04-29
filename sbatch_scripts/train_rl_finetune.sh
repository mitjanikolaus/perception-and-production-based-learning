#!/bin/bash
#
#SBATCH --job-name=RL
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_rl_finetune.out

source activate egg

python -u train.py --lr 0.00001 --max_len 20 --freeze-receiver --receiver-checkpoint ~/data/visual_ref/checkpoints/ranking_fine_tune_resnet/ranking.pt --sender-checkpoint ~/data/visual_ref/checkpoints/captioning/interactive.pt --print-sample-interactions --sender functional --log-frequency 50 --eval-frequency 300 --weight-structural-loss 0 --out-checkpoints-dir out_rl_finetune --sender-entropy-coeff 0.1 # --eval-semantics


