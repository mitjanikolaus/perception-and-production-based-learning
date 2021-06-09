#!/bin/bash
#
#SBATCH --job-name=RL-ST
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/rl_captioning_st_trainset_frac_1.0_length_cost_0.01_entropy_coeff_0.1.out

source activate egg
python -u fine_tune_image_captioning.py --seed 7 --lr 0.00001 --log-frequency 50 --n-epochs 100 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir out_rl_captioning_st_trainset_frac_1.0_length_cost_0.01_entropy_coeff_0.1/ --entropy-coeff 0.1 --length-cost 0.01 --training-set-size 1.0 --weight-supervised-loss 0

