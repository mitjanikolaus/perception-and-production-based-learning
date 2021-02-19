from __future__ import print_function

import argparse
import pickle
import os

import h5py
import imageio
from skimage.transform import resize

import torch
import torch.distributions
import torch.utils.data

import egg.core as core
from dataset import CaptionDataset, SyntaxEvalDataset
from models import ImageCaptioner
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH, show_image,
)
from train_image_captioning import (
    CHECKPOINT_PATH_IMAGE_CAPTIONING_BEST,
)
from utils import decode_caption


def main(args):
    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    images = h5py.File(
        os.path.join(DATA_PATH, IMAGES_FILENAME["test"]), "r"
    )

    # Load captions
    with open(os.path.join(DATA_PATH, CAPTIONS_FILENAME["test"]), "rb") as file:
        captions = pickle.load(file)

    for img_id, image in images.items():
        # print(img_id)
        for caption in captions[int(img_id)]:
            decoded_caption = decode_caption(caption, vocab)
            if "jenny" in decoded_caption and "mike" in decoded_caption:
                if (not "mike and jenny" in decoded_caption) and (not "jenny and mike" in decoded_caption):
                    distractor = decoded_caption.replace("jenny","XXXX").replace("mike","jenny").replace("XXXX","mike")
                    print(f"{img_id},{decoded_caption},{distractor}")
                    show_image(image)

        # img_distractor = imageio.imread(f"data/{img_id}.png")
        #
        # # discard transparency channel
        # img_distractor = img_distractor[..., :3]
        #
        # # downscale to 224x224 pixes (optimized for resnet)
        # img_distractor = resize(img_distractor, (224, 224), preserve_range=True).astype("uint8")
        #
        # show_image(img_distractor)


def get_args():
    parser = argparse.ArgumentParser()

    return core.init(parser)


if __name__ == "__main__":
    args = get_args()
    main(args)
