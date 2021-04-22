import argparse
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from utils import DEFAULT_BATCH_SIZE

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

LEGEND_GROUPED_NOUNS = {
    "nouns": "nouns",
    "data/semantics_eval_verbs.csv": "verbs",
    "data/semantics_eval_adjectives.csv": "adjectives",
    "data/semantics_eval_adjective_noun_binding.csv": "adjective-noun dependency",
    "data/semantics_eval_verb_noun_binding_filtered.csv": "verb-noun dependency",
    "data/semantics_eval_semantic_roles_filtered.csv": "semantic roles",
}


def main(args):
    sns.set_context(
        "paper",
        rc={
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        },
    )

    all_scores = []
    for run, scores_file in enumerate(args.scores_files):
        scores = pd.read_csv(scores_file)
        for column_name in scores.columns:
            if not column_name == "epoch" or column_name == "batch_id":
                scores[column_name] = (
                    scores[column_name].rolling(args.rolling_window, min_periods=1).mean()
                )

        scores["num_samples"] = scores.aggregate(lambda x: (x["epoch"] + 1) * x["batch_id"] * DEFAULT_BATCH_SIZE, axis=1)

        print(
            f"Epoch with min val loss:{scores[scores['val_loss'] == scores['val_loss'].min()]['epoch'].values[0]}"
        )
        print(
            f"num_samples with min val loss:{scores[scores['val_loss'] == scores['val_loss'].min()]['num_samples'].values[0]}"
        )

        scores.set_index("num_samples", inplace=True)

        legend = LEGEND
        scores.rename(columns=legend, inplace=True)
        if args.group_noun_accuracies:
            scores.insert(
                0, "nouns", scores[["persons", "animals", "objects"]].mean(axis=1)
            )
            legend = LEGEND_GROUPED_NOUNS

        all_scores.append(scores.copy())

    all_scores = pd.concat(all_scores)

    sns.lineplot(data=all_scores[list(legend.values())], ci="sd")

    plt.xlim((0, args.x_lim))
    plt.ylim((0.49, args.y_lim))

    # Add chance level line
    plt.axhline(y=0.5, color="black", label="Chance level", linestyle="--")
    plt.text(440000, 0.507, "Chance level", fontsize=12, va="center", ha="center")

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
        "--x-lim", default=TRAINING_SET_SIZE * 15, type=int,
    )
    parser.add_argument(
        "--y-lim", default=1.0, type=float,
    )
    parser.add_argument(
        "--group-noun-accuracies", default=False, action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
