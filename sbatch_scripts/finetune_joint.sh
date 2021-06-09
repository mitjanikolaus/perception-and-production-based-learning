#!/bin/bash
#
#SBATCH --job-name=finetune-joint
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/fine_tune_joint_length_cost_0_entropy_coeff_0_supervised_weight_0.1.out

source activate egg
python -u fine_tune_image_captioning.py --seed 7 --lr 0.00001 --log-frequency 100 --n-epochs 25 --model joint --fine-tune-resnet --checkpoint ~/data/visual_ref/checkpoints/captioning/joint.pt --out-checkpoints-dir out_fine_tune_joint_length_cost_0_entropy_coeff_0_supervised_weight_0.1 --entropy-coeff 0 --length-cost 0 --weight-supervised-loss 0.1 --eval-semantics


