#!/bin/bash
#
#SBATCH --job-name=finetune-ST
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=48000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/fine_tune_captioning_st_trainset_frac_0.7_frequency_rl_updates_10.out

source activate egg
#python -u train_image_captioning.py --seed 1 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_1/xsl_then_alternate/1/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/1/show_and_tell_train_frac_0.05.pt --training-set-size 0.05

#python -u train_image_captioning.py --seed 2 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_1/xsl_then_alternate/2/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/2/show_and_tell_train_frac_0.05.pt --training-set-size 0.05

#python -u train_image_captioning.py --seed 3 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_1/xsl_then_alternate/3/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/3/show_and_tell_train_frac_0.05.pt --training-set-size 0.05

#python -u train_image_captioning.py --seed 4 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_1/xsl_then_alternate/4/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/4/show_and_tell_train_frac_0.05.pt --training-set-size 0.05

python -u train_image_captioning.py --seed 1 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_amount_of_pretraining/xsl_then_alternate/0.7/1/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/1/show_and_tell_train_frac_0.7.pt --training-set-size 1.0

python -u train_image_captioning.py --seed 2 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_amount_of_pretraining/xsl_then_alternate/0.7/2/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/2/show_and_tell_train_frac_0.7.pt --training-set-size 1.0

python -u train_image_captioning.py --seed 3 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_amount_of_pretraining/xsl_then_alternate/0.7/3/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/3/show_and_tell_train_frac_0.7.pt --training-set-size 1.0

python -u train_image_captioning.py --seed 4 --lr 0.00001 --log-frequency 100 --n-epochs 5000 --model show_and_tell --fine-tune-resnet --out-checkpoints-dir results_amount_of_pretraining/xsl_then_alternate/0.7/4/ --eval-semantics --frequency-rl-updates 10 --entropy-coeff 0 --checkpoint ~/nn_language_acquisition/visual_ref/results_1/xsl/4/show_and_tell_train_frac_0.7.pt --training-set-size 1.0

