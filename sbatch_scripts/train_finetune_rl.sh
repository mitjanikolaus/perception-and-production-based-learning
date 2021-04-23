#!/bin/bash
#
#SBATCH --job-name=RL
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_finetune_rl.out

source activate egg
python -u train.py --lr 0.00001 --n_epochs 50 --batch_size 32 --max_len 25 --freeze-receiver --receiver-checkpoint ~/data/visual_ref/checkpoints/ranking_run_3/ranking.pt --sender-checkpoint ~/data/visual_ref/checkpoints/captioning/interactive.pt --print-sample-interactions --sender functional --log-frequency 100 --eval-frequency 200

