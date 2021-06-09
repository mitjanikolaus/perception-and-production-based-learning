#!/bin/bash
#
#SBATCH --job-name=finetune-ST
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=48000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCHCOMMENT --constraint=cuda75|cuda61
#SBATCH --output=out/fine_tune_captioning_st_trainset_frac_1.0_frequency_rl_updates_-1.out

source activate egg
python -u train_image_captioning.py --seed 4 --lr 0.00001 --log-frequency 50 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results/xsl_then_rl/4/ --eval-semantics --frequency-rl-updates -1 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results/xsl/4/show_and_tell_train_frac_1.0.pt --training-set-size 1.0
