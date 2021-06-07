import argparse
import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from utils import LEGEND_GROUPED_NOUNS, LEGEND

import numpy as np


def main(args):
    # sns.set_context(
    #     "paper",
    #     rc={
    #         "font.size": 12,
    #         "axes.titlesize": 12,
    #         "axes.labelsize": 12,
    #         "xtick.labelsize": 12,
    #         "ytick.labelsize": 12,
    #         "legend.fontsize": 12,
    #     },
    # )

    legend = LEGEND
    if args.group_noun_accuracies:
        legend = LEGEND_GROUPED_NOUNS

    all_scores = []
    for root, dirs, files in os.walk(args.results_folder):
        for name in files:
            scores_file = os.path.join(root, name)
            if scores_file.endswith(".csv"):
                scores = pd.read_csv(scores_file)

                scores.set_index("num_samples", inplace=True)

                scores.rename(columns=LEGEND, inplace=True)
                if args.group_noun_accuracies:
                    scores.insert(
                        0,
                        "nouns",
                        scores[["persons", "animals", "objects"]].mean(axis=1),
                    )

                metric = "bleu_score_val"
                best_score = (
                    scores[scores[metric] == scores[metric].max()].iloc[0].to_dict()
                )
                if "epoch" in scores.columns:
                    print(f"Epoch with max {metric}:{best_score['epoch']}")

                print("Top scores:")
                for name in LEGEND.values():
                    print(f"Accuracy for {name}: {best_score[name]:.3f}")

                overall_average = np.mean(
                    [best_score[name] for name in LEGEND.values()]
                )
                print(f"Overview Average: {overall_average:.3f}")
                best_score["average"] = overall_average

                # Read train set frac information from file name
                best_score["train_frac"] = float(
                    str(scores_file.split("train_frac_")[1]).split("_accuracies")[0]
                )

                # Read train setup information from folder name
                best_score["setup"] = scores_file.split("/")[-2]

                if args.print_per_task_accs:
                    for name in legend.values():
                        all_scores.append({"setup": best_score["setup"], "train_frac": best_score["train_frac"], "task": name, "accuracy": best_score[name]})

                else:
                    all_scores.append({"setup": best_score["setup"], "train_frac": best_score["train_frac"], "accuracy": best_score["average"]})

    all_scores = pd.DataFrame(all_scores)


    all_scores.set_index("train_frac", inplace=True)

    hue = "task" if args.print_per_task_accs else "setup"
    style = "setup"
    if all_scores.setup.unique().size == 1:
        style = "task"
    g = sns.lineplot(
        data=all_scores,
        x="train_frac",
        y="accuracy",
        hue=hue,
        markers=True,
        style=style,
        err_style="bars",
    )
    # g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3)
    plt.subplots_adjust(top=0.7)


    plt.ylim((0.4, args.y_lim))

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
    parser.add_argument(
        "--print-per-task-accs", default=False, action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
