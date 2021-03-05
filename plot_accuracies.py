
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
    "data/semantics_eval_adjective_noun_binding.csv": "adjective-noun dependency",
    "data/semantics_eval_verb_noun_binding_filtered.csv": "verb-noun dependency",
    "data/semantics_eval_semantic_roles_filtered.csv": "semantic roles",
}


def main(args):
    all_scores = []
    for run, scores_file in enumerate(args.scores_files):
        scores = pickle.load(open(scores_file, "rb"))
        scores = pd.DataFrame(scores)
        for column_name in scores.columns:
            scores[column_name] = scores[column_name].rolling(args.rolling_window, min_periods=1).mean()

        # Delete superfluous logging entries (these mess up epoch calculation otherwise)
        epoch = TRAINING_SET_SIZE
        to_delete = []
        for i, row in scores.iterrows():
            if row.name * DEFAULT_BATCH_SIZE * DEFAULT_LOG_FREQUENCY > epoch:
                epoch += TRAINING_SET_SIZE
                to_delete.append(row.name)
        scores.drop(labels=to_delete, inplace=True)
        scores.reset_index(drop=True, inplace=True)

        scores["num_samples"] = scores.index.map(lambda x: (x * DEFAULT_BATCH_SIZE * DEFAULT_LOG_FREQUENCY))
        scores["epoch"] = scores.index.map(lambda x: (x * DEFAULT_BATCH_SIZE * DEFAULT_LOG_FREQUENCY) / TRAINING_SET_SIZE)

        print(f"Epoch with min val loss:{scores[scores['val_loss'] == scores['val_loss'].min()]['epoch'].values[0]}")
        print(f"num_samples with min val loss:{scores[scores['val_loss'] == scores['val_loss'].min()]['num_samples'].values[0]}")
        del scores["epoch"]
        del scores["val_loss"]

        scores.set_index("num_samples", inplace=True)

        scores.rename(columns=LEGEND, inplace=True)

        all_scores.append(scores.copy())

    all_scores = pd.concat(all_scores)

    sns.lineplot(data=all_scores, ci="sd")

    plt.xlim((0, args.x_lim))

    # Add chance level line
    plt.axhline(y=0.5, color="black", label="Chance level", linestyle="--")
    plt.text(450000, 0.505, 'Chance level', fontsize=10, va='center', ha='center')

    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-files", type=str, nargs="+", required=True,
    )
    parser.add_argument(
        "--rolling-window", default=30, type=int,
    )
    parser.add_argument(
        "--x-lim", default=TRAINING_SET_SIZE*10, type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
