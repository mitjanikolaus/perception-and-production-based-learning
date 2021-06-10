import argparse
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from train_image_captioning import UNIQUE_VERBS
from utils import LEGEND_GROUPED_NOUNS, LEGEND

import numpy as np

LEGEND_STATS_SENTENCE_LENGTH = {
    "seq_lengths": "sentence_length",
}

LEGEND_STATS_PERSONS = {
    "jenny_occurrences": "jenny",
    "mike_occurrences": "mike",
}

LEGEND_STATS_VERBS = {verb: verb for verb in UNIQUE_VERBS}

LEGEND_STATS_ALL = {
    **LEGEND_STATS_SENTENCE_LENGTH,
    **LEGEND_STATS_PERSONS,
    **LEGEND_STATS_VERBS,
}


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

    all_scores = []

    for run, scores_file in enumerate(args.scores_files):
        scores = pd.read_csv(scores_file)
        for column_name in scores.columns:
            if not (column_name == "epoch" or column_name == "batch_id"):
                scores[column_name] = (
                    scores[column_name]
                    .rolling(args.rolling_window, min_periods=1)
                    .mean()
                )

        scores.set_index("num_samples", inplace=True)

        scores.rename(columns=LEGEND_STATS_ALL, inplace=True)

        metric = (
            "bleu_score_val"
            if "bleu_score_val" in scores.columns
            else "bleu_score_train"
        )
        best_score = scores[scores[metric] == scores[metric].max()]
        if len(best_score) == 0:
            print("No best score, taking last value!")
            best_score = scores.tail(1)
        if "epoch" in scores.columns:
            print(f"Epoch with max {metric}:{best_score['epoch'].values[0]}")

        print(f"num_samples with max {metric}:{best_score.index.values[0]}")

        # overall_average = np.mean([best_score[name].values[0] for name in LEGEND.values()])
        # print(f"Overview Average: {overall_average:.3f}")

        # Read train setup information from file name
        setup = str(scores_file.split("/")[7])

        scores_setup = []
        for row in scores.iterrows():
            for name in LEGEND_STATS_ALL.values():
                filtered_scores = {}
                filtered_scores["score"] = name
                filtered_scores["value"] = row[1][name]
                filtered_scores["num_samples"] = row[0]
                filtered_scores["setup"] = setup
                scores_setup.append(filtered_scores)

        all_scores.extend(scores_setup.copy())

    fig, axes = plt.subplots(2, 2, sharey="row", sharex="all")  # figsize=(15, 5)

    all_scores = pd.DataFrame(all_scores)

    for axis_y, setup in enumerate(all_scores.setup.unique()):
        for axis_x, legend in enumerate(
            [LEGEND_STATS_PERSONS, LEGEND_STATS_VERBS]
        ):
            legend_values = legend.values()
            scores_setup = all_scores[(all_scores.setup == setup) & (all_scores.score.isin(legend_values))]
            legend = True if axis_y == 1 else False
            g = sns.lineplot(
                ax=axes[axis_x][axis_y],
                data=scores_setup,
                x="num_samples",
                y="value",
                hue="score",
                # style="score",
                legend=legend,
            )
            if axis_x == 1:
                axes[axis_x][axis_y].set_ylabel("Occurrences/Sentence")
            else:
                axes[axis_x][axis_y].set_ylabel("Occurrences/Sentence")
            if axis_y == 1:
                g.legend(loc='best', fontsize=9, ncol=2)
            if axis_x == 1:
                plt.ylim((0, 0.15))

    axes[0][0].set_title(f"{all_scores.setup.unique()[0]}")
    axes[0][1].set_title(f"{all_scores.setup.unique()[1]}")

    # plt.ylabel("Occurrences/Sentence")

    if args.x_lim:
        plt.xlim((0, args.x_lim))
    # plt.ylim((0.49, args.y_lim))

    # plt.xlabel("num_samples")

    # plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


    fig, axes = plt.subplots(1, 2, sharey="row", sharex="row")  # figsize=(15, 5)

    all_scores = pd.DataFrame(all_scores)

    for axis_y, setup in enumerate(all_scores.setup.unique()):
        legend = LEGEND_STATS_SENTENCE_LENGTH
        legend_values = legend.values()
        scores_setup = all_scores[(all_scores.setup == setup) & (all_scores.score.isin(legend_values))]
        sns.lineplot(
            ax=axes[axis_y],
            data=scores_setup,
            x="num_samples",
            y="value",
            hue="score",
            # style="score",
            legend=False,
        )
        axes[axis_y].set_ylabel("Mean Sentence Length")

    axes[0].set_title(f"{all_scores.setup.unique()[0]}")
    axes[1].set_title(f"{all_scores.setup.unique()[1]}")

    if args.x_lim:
        plt.xlim((0, args.x_lim))

    plt.tight_layout()
    plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-files", type=str, nargs="+", required=True,
    )
    parser.add_argument(
        "--rolling-window", default=1, type=int,
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
