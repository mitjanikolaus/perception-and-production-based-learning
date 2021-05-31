#!/bin/bash
#
#SBATCH --job-name=finetune-ST
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/fine_tune_captioning_st_length_cost_.01.out

source activate egg
python -u fine_tune_image_captioning.py --seed 7 --lr 0.000001 --log-frequency 100 --n-epochs 25 --model show_and_tell --fine-tune-resnet --checkpoint ~/data/visual_ref/checkpoints/captioning/show_and_tell.pt --out-checkpoints-dir out_fine_tune_captioning_st_length_cost_.01 --entropy-coeff 0.01 --length-cost 0.01 --eval-semantics

