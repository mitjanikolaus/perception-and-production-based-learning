
import argparse
import pickle
import pandas as pd
import numpy as np
import seaborn as sns


import matplotlib.pyplot as plt

from preprocess import DATASET_SIZE
from utils import DEFAULT_LOG_FREQUENCY, DEFAULT_BATCH_SIZE

TRAINING_SET_SIZE = 48198

LEGEND = {
    "data/semantics_eval_persons.csv": "persons",
    "data/semantics_eval_animals.csv": "animals",
    "data/semantics_eval_inanimates.csv": "objects",
    "data/semantics_eval_verbs.csv": "verbs",
    "data/semantics_eval_adjectives.csv": "adjectives",
    "data/semantics_eval_adjective_noun_binding.csv": "adjective-noun binding",
    "data/semantics_eval_verb_noun_binding_filtered.csv": "verb-noun binding",
    "data/semantics_eval_semantic_roles_filtered.csv": "semantic roles",
}

def main(args):
    scores = pickle.load(open(args.scores_file, "rb"))
    scores = pd.DataFrame(scores)
    for column_name in scores.columns:
        scores[column_name] = scores[column_name].rolling(args.rolling_window, min_periods=1).mean()

    scores["epoch"] = scores.index.map(lambda x: (x * DEFAULT_BATCH_SIZE * DEFAULT_LOG_FREQUENCY) / TRAINING_SET_SIZE)

    scores.set_index("epoch", inplace=True)

    del scores["val_loss"]

    scores.rename(columns=LEGEND, inplace=True)

    sns.lineplot(data=scores)

    plt.xlim((0, args.x_lim))
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-file", type=str,
    )
    parser.add_argument(
        "--rolling-window", default=100, type=int,
    )
    parser.add_argument(
        "--x-lim", default=10, type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
