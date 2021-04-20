# cross-situational-learning-abstract-scenes

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

## Training and Evaluation

The training script can be used to train the ranking model and evaluate it every $LOG_FREQ iterations (default: 100).
Checkpoints and accuracies are saved to $CHECKPOINT_DIR.

```
python train_image_sentence_ranking.py --checkpoint-dir $CHECKPOINT_DIR --log-frequency $LOG_FREQ
```

## Plot learning trajectory

The learning trajectory can be plotted using the following command:

```
python plot_accuracies.py --group-noun-accuracies --scores-files $CHECKPOINT_DIR/ranking_accuracies.p
```

If multiple scores files are given (e.g. for different model runs), mean and standard deviation are automatically
computed and shown in the plot.
