#!/bin/bash
#
#SBATCH --job-name=multi
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_multitask.out

source activate egg
python -u train.py --lr 0.0001 --n_epochs 50 --batch_size 32 --max_len 25 --freeze-receiver --receiver-checkpoint ~/data/visual_ref/checkpoints/ranking_run_3/ranking.pt --print-sample-interactions --sender functional --log-frequency 50

