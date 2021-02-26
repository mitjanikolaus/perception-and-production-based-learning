
import argparse
import pickle
import pandas as pd
import numpy as np
import seaborn as sns

import egg.core as core

import matplotlib.pyplot as plt

from utils import SEMANTIC_ACCURACIES_PATH_IMAGE_CAPTIONING


def main(args):
    scores = pickle.load(open(args.scores_file, "rb"))
    scores = pd.DataFrame(scores)
    for column_name in scores.columns:
        scores[column_name] = scores[column_name].rolling(args.rolling_window, center=True).mean()
    sns.lineplot(data=scores)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-file", default=SEMANTIC_ACCURACIES_PATH_IMAGE_CAPTIONING, type=str,
    )
    parser.add_argument(
        "--rolling-window", default=10, type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
