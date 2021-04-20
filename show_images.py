#  python -m train --vocab_size=10 --n_epochs=15 --random_seed=7 --lr=1e-3 --batch_size=32 --optimizer=adam
import argparse
import os

import h5py
import torch

from preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    show_image,
)

import pandas as pd
import matplotlib.pyplot as plt


def main(args):

    images = h5py.File(os.path.join(DATA_PATH, IMAGES_FILENAME[args.split]), "r")

    data = pd.read_csv(args.eval_csv)

    for i, row in data.iterrows():
        img_id, target_sentence, distractor_sentence, agent_left = row

        image_data = images[str(img_id)][()]

        image = torch.FloatTensor(image_data)

        features_scale_factor = 255
        image = image / features_scale_factor

        plt.title(img_id)
        show_image(image)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="test", type=str, help="dataset split to use",
    )
    parser.add_argument(
        "--eval-csv", type=str, required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
