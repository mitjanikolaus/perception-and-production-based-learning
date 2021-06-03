import argparse
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from utils import LEGEND_GROUPED_NOUNS, LEGEND


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
            if not (column_name == "epoch" or column_name == "batch_id"):
                scores[column_name] = (
                    scores[column_name].rolling(args.rolling_window, min_periods=1).mean()
                )

        scores.set_index("num_samples", inplace=True)

        legend = LEGEND
        scores.rename(columns=legend, inplace=True)
        if args.group_noun_accuracies:
            scores.insert(
                0, "nouns", scores[["persons", "animals", "objects"]].mean(axis=1)
            )
            legend = LEGEND_GROUPED_NOUNS

        metric = "bleu_score_val" if "bleu_score_val" in scores.columns else "bleu_score_train"
        best_score = scores[scores[metric] == scores[metric].max()]
        if len(best_score) == 0:
            print("No best score, taking last value!")
            best_score = scores.tail(1)
        if "epoch" in scores.columns:
            print(
                f"Epoch with max {metric}:{best_score['epoch'].values[0]}"
            )

        print(
            f"num_samples with max {metric}:{best_score.index.values[0]}"
        )

        print("Top scores:")
        for name in LEGEND.values():
            print(f"Accuracy for {name}: {best_score[name].values[0]:.3f}")


        all_scores.append(scores.copy())

    all_scores = pd.concat(all_scores)

    if "val_acc" in all_scores.columns:
        print(f"Max val acc: {scores['val_acc'].max()}")
        legend["val_acc"] = "val_acc"

    if "bleu_score_train" in all_scores.columns:
        legend["bleu_score_train"] = "bleu_score_train"

    if "bleu_score_val" in all_scores.columns:
        legend["bleu_score_val"] = "bleu_score_val"

    if "val_loss" in all_scores.columns:
        legend["val_loss"] = "val_loss"

    sns.lineplot(data=all_scores[list(legend.values())], ci="sd")

    if args.x_lim:
        plt.xlim((0, args.x_lim))
    plt.ylim((0.49, args.y_lim))

    # Add chance level line
    plt.axhline(y=0.5, color="black", label="Chance level", linestyle="--")
    plt.text(0, 0.507, "Chance level", fontsize=12, va="center", ha="center")

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
        "--x-lim", default=None, type=int,
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
