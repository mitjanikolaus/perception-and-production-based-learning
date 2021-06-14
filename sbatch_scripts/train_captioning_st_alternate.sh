#!/bin/bash
#
#SBATCH --job-name=alternate
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/train_st_alternate_trainset_frac_1.0_frequency_rl_updates_10.out

source activate egg
#python -u train_image_captioning.py --seed 7 --lr 0.0001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --out-checkpoints-dir results/alternate_new/0/ --training-set-size 1.0 --fine-tune-resnet --entropy-coeff 0 --eval-semantics --frequency-rl-updates 10

#python -u train_image_captioning.py --seed 1 --lr 0.0001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --out-checkpoints-dir results/alternate_new/1/ --training-set-size 1.0 --fine-tune-resnet --entropy-coeff 0 --eval-semantics --frequency-rl-updates 10

#python -u train_image_captioning.py --seed 2 --lr 0.0001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --out-checkpoints-dir results/alternate_new/2/ --training-set-size 1.0 --fine-tune-resnet --entropy-coeff 0 --eval-semantics --frequency-rl-updates 10

#python -u train_image_captioning.py --seed 3 --lr 0.0001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --out-checkpoints-dir results/alternate_new/3/ --training-set-size 1.0 --fine-tune-resnet --entropy-coeff 0 --eval-semantics --frequency-rl-updates 10

python -u train_image_captioning.py --seed 4 --lr 0.0001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --out-checkpoints-dir results/alternate_new/4/ --training-set-size 1.0 --fine-tune-resnet --entropy-coeff 0 --eval-semantics --frequency-rl-updates 10


