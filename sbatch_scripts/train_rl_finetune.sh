#!/bin/bash
#
#SBATCH --job-name=RL-finetune
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_rl_finetune_multiple_receivers_struct_loss_1.out

source activate egg

python -u train.py --lr 0.00001 --random_seed 7 --max_len 15 --freeze-receiver --receiver-checkpoint ~/data/visual_ref/checkpoints/ranking_run_1/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_2/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_3/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_4/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_0/ranking.pt --sender-checkpoint ~/data/visual_ref/checkpoints/captioning/interactive.pt --print-sample-interactions --sender functional --log-frequency 100 --eval-frequency 500 --weight-structural-loss 1 --out-checkpoints-dir out_rl_finetune --sender-entropy-coeff 0.1 --eval-semantics


