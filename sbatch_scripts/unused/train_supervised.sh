#!/bin/bash
#
#SBATCH --job-name=Superv
#
#SBATCH --ntasks=1
#SBATCH --time=32:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/train_supervised_max_len_10_vocab_1057_lr-5_batch_size_32_receivers_1.out

source activate egg
python -u train.py --lr 0.00001 --random_seed 7 --max_len 10 --freeze-receiver --receiver-checkpoints ~/data/visual_ref/checkpoints/ranking_fine_tune_resnet/ranking.pt --print-sample-interactions --sender functional --log-frequency 1000 --eval-frequency 5000 --weight-structural-loss 1 --weight-functional-loss 0 --out-checkpoints-dir out_supervised --batch_size 32 --eval-semantics 


# ~/data/visual_ref/checkpoints/ranking_run_1/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_2/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_3/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_4/ranking.pt ~/data/visual_ref/checkpoints/ranking_run_0/ranking.pt
