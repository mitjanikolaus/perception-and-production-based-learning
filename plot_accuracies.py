
import argparse
import pickle
import pandas as pd
import numpy as np
import seaborn as sns

import egg.core as core

import matplotlib.pyplot as plt

from utils import SEMANTIC_ACCURACIES_PATH_IMAGE_CAPTIONING


def main(args):
    scores = pickle.load(open(SEMANTIC_ACCURACIES_PATH_IMAGE_CAPTIONING, "rb"))
    scores = pd.DataFrame(scores)
    sns.lineplot(data=scores)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-file", default=SEMANTIC_ACCURACIES_PATH_IMAGE_CAPTIONING, type=str,
    )

    return core.init(parser)


if __name__ == "__main__":
    args = get_args()
    main(args)
