import argparse
import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from utils import LEGEND_GROUPED_NOUNS, LEGEND

import numpy as np


def main(args):
    all_accuracies = {key: [] for key in LEGEND.values()}
    all_accuracies["average"] = []

    for root, dirs, files in os.walk(args.results_folder):
        for name in files:
            scores_file = os.path.join(root, name)
            if scores_file.endswith(".csv"):
                scores = pd.read_csv(scores_file)

                scores.set_index("num_samples", inplace=True)

                scores.rename(columns=LEGEND, inplace=True)

                metric = "bleu_score_val"
                best_score = (
                    scores[scores[metric] == scores[metric].max()].iloc[0].to_dict()
                )

                overall_average = np.mean(
                    [best_score[name] for name in LEGEND.values()]
                )
                print(f"Run Overview Average: {overall_average:.3f}")
                best_score["average"] = overall_average

                for name in all_accuracies.keys():
                    all_accuracies[name].append(best_score[name])

    print("\n")

    for name in all_accuracies.keys():
        print(f"{name}: {np.mean(all_accuracies[name]):.2f} \pm {np.std(all_accuracies[name]):.2f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder", type=str, required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
