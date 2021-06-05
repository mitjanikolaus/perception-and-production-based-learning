import argparse
import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from utils import LEGEND_GROUPED_NOUNS, LEGEND

import numpy as np


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

    legend = LEGEND
    if args.group_noun_accuracies:
        legend = LEGEND_GROUPED_NOUNS

    all_scores = []
    for root, dirs, files in os.walk(args.results_folder):
        for name in files:
            scores_file = os.path.join(root, name)
            scores = pd.read_csv(scores_file)

            scores.set_index("num_samples", inplace=True)

            scores.rename(columns=legend, inplace=True)
            if args.group_noun_accuracies:
                scores.insert(
                    0, "nouns", scores[["persons", "animals", "objects"]].mean(axis=1)
                )

            metric = "bleu_score_val"
            best_score = scores[scores[metric] == scores[metric].max()]
            if "epoch" in scores.columns:
                print(
                    f"Epoch with max {metric}:{best_score['epoch'].values[0]}"
                )

            print("Top scores:")
            for name in LEGEND.values():
                print(f"Accuracy for {name}: {best_score[name].values[0]:.3f}")

            overall_average = np.mean([best_score[name].values[0] for name in LEGEND.values()])
            print(f"Overview Average: {overall_average:.3f}")
            best_score["average"] = overall_average

            # Read train set frac information from file name
            best_score["train_frac"] = float(str(scores_file.split("train_frac_")[1]).split("_accuracies")[0])

            # Read train setup information from folder name
            best_score["setup"] = scores_file.split("/")[-2]

            all_scores.append(best_score.copy())

    all_scores = pd.concat(all_scores)

    legend["train_frac"] = ["train_frac"]

    sns.scatterplot(data=all_scores, x="train_frac", y="average", hue="setup")

    plt.ylim((0.49, args.y_lim))

    # Add chance level line
    plt.axhline(y=0.5, color="black", label="Chance level", linestyle="--")
    plt.text(0.85, 0.507, "Chance level", fontsize=12, va="center", ha="center")

    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder", type=str, required=True,
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
