
import argparse
import pickle
import pandas as pd
import numpy as np
import seaborn as sns


import matplotlib.pyplot as plt

from utils import DEFAULT_LOG_FREQUENCY, DEFAULT_BATCH_SIZE


def main(args):
    scores = pickle.load(open(args.scores_file, "rb"))
    scores = pd.DataFrame(scores)
    for column_name in scores.columns:
        scores[column_name] = scores[column_name].rolling(args.rolling_window, min_periods=1).mean()

    scores["num_samples"] = scores.index.map(lambda x: x * DEFAULT_BATCH_SIZE * DEFAULT_LOG_FREQUENCY)
    scores.set_index("num_samples", inplace=True)

    sns.lineplot(data=scores)
    plt.xlim((0, args.x_lim))
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-file", type=str,
    )
    parser.add_argument(
        "--rolling-window", default=10, type=int,
    )
    parser.add_argument(
        "--x-lim", default=400000, type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
