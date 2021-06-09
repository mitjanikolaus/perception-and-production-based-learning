import argparse
import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from utils import LEGEND_GROUPED_NOUNS, LEGEND

import numpy as np


def main(args):
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
                    print(f"Epoch with max {metric}:{best_score['epoch']} score: {best_score[metric]:.3f}")

                print("Top scores:")
                for name in LEGEND.values():
                    print(f"Accuracy for {name}: {best_score[name]:.2f}")

                overall_average = np.mean(
                    [best_score[name] for name in LEGEND.values()]
                )
                print(f"Overview Average: {overall_average:.2f}")
                best_score["average"] = overall_average

                # Read train set frac information from file name
                best_score["freq_rl_updates"] = int(str(scores_file.split("frequency_rl_updates_")[1]).split("_")[0])

                score = {"freq_rl_updates": best_score["freq_rl_updates"]}
                for name in legend.values():
                    score[name] = best_score[name]
                score["average"] = best_score["average"]
                all_scores.append(score)

    all_scores = pd.DataFrame(all_scores)
    print(all_scores.T.to_latex(float_format="%.3f"))


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
