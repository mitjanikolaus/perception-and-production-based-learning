# perception-and-production-based-learning

## Environment
An anaconda environment can be setup by using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate xsl
```

## Preprocessing
Download the data from the [abstract scenes dataset](https://vision.ece.vt.edu/clipart/) and extract the files.

Afterwards, preprocess the images and sentences:
```
python preprocess.py --dataset-folder $PATH_TO_ABSTRACT_SCENES_DATA
```

## Model Training and Evaluation

The training script can be used to train the ranking model and evaluate it every $LOG_FREQ iterations (default: 100).
Checkpoints and semantic accuracies are saved to $CHECKPOINT_DIR. The frequency of production-based learning updates
(reinforcement learning updates) can be modulated using $FREQ_RL_UPDATES (set to -1 for only RL, and to 0 for
pure supervised learning).
```
python train_image_captioning.py --log-frequency $LOG_FREQ --out-checkpoints-dir $CHECKPOINT_DIR --frequency-rl-updates $FREQ_RL_UPDATES --fine-tune-resnet --eval-semantics --model show_and_tell
```

If you want to continue training from an existing checkpoint (e.g., for fine-tuning), you can indicate this checkpoint
using the parameter ```--checkpoint $PATH_TO_CHECKPOINT```.



## Acknowledgements
This repo was built based on https://github.com/mitjanikolaus/cross-situational-learning-abstract-scenes

