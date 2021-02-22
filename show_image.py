#  python -m train --vocab_size=10 --n_epochs=15 --random_seed=7 --lr=1e-3 --batch_size=32 --optimizer=adam
import argparse
import os

import h5py
import torch

import egg.core as core
from preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME, show_image,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    images = h5py.File(
        os.path.join(DATA_PATH, IMAGES_FILENAME[args.split]), "r"
    )
    image_data = images[str(args.image_id)][()]

    image = torch.FloatTensor(image_data)

    features_scale_factor = 255
    image = image / features_scale_factor

    show_image(image)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="dataset split to use",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        required=True,
    )
    args = core.init(parser)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
