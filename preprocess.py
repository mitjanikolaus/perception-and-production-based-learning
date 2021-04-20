"""Preprocess the abstract scenes images and captions and store them in a hdf5 file"""

import argparse
import os
import pickle
import string
import sys

from collections import Counter
import imageio

import matplotlib.pyplot as plt

import h5py
import nltk
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from tqdm import tqdm

import numpy as np

VOCAB_FILENAME = "vocab.p"
IMAGES_FILENAME = {
    "train": "images_train.hdf5",
    "val": "images_val.hdf5",
    "test": "images_test.hdf5",
}
CAPTIONS_FILENAME = {
    "train": "captions_train.p",
    "val": "captions_val.p",
    "test": "captions_test.p",
}

RANDOM_SEED = 1

CAPTIONS_PER_IMAGE = 6

DATA_PATH = os.path.expanduser("~/data/abstract_scenes/preprocessed/")
DATASET_SIZE = 10020

MEAN_ABSTRACT_SCENES = [107.36, 177.26, 133.54]
STD_ABSTRACT_SCENES = [49.11, 45.28, 75.83]

MAX_CAPTION_LEN = 25
TOKEN_PADDING = "<pad>"
TOKEN_START = "<sos>"
TOKEN_END = "<eos>"

VAL_SET_SIZE = 0.1


def encode_caption(caption, vocab):
    return (
        [vocab.stoi[TOKEN_START]]
        + [vocab.stoi[word] for word in caption]
        + [vocab.stoi[TOKEN_END]]
    )


def encode_captions(captions, vocab):
    return [encode_caption(caption, vocab) for caption in captions]


def show_image(img_data):
    plt.imshow(img_data), plt.axis("off")
    plt.show()


def preprocess_images_and_captions(
    dataset_folder, output_folder, vocab_min_freq,
):
    images = []
    word_freq = Counter()

    images_folder = os.path.join(dataset_folder, "RenderedScenes")
    for scene_id in tqdm(range(1002)):
        for image_scene_id in range(10):
            img_filename = f"Scene{scene_id}_{image_scene_id}.png"
            img_path = os.path.join(images_folder, img_filename)

            img = imageio.imread(img_path)

            # discard transparency channel
            img = img[..., :3]

            # downscale to 224x224 pixes (optimized for resnet)
            img = resize(img, (224, 224), preserve_range=True)

            # show_image(img / 255)
            images.append(img)

    # Calculate Mean and Standard deviation of images for sample
    images_analysis = np.array(images[:500])
    print("Mean and standard deviation: ")
    print(np.mean(images_analysis.reshape(-1, 3), axis=0))
    print(np.std(images_analysis.reshape(-1, 3), axis=0))

    captions_file_1 = os.path.join(
        dataset_folder, "SimpleSentences", "SimpleSentences1_10020.txt"
    )
    captions_file_2 = os.path.join(
        dataset_folder, "SimpleSentences", "SimpleSentences2_10020.txt"
    )

    captions = {}
    for image_id in range(DATASET_SIZE):
        captions[image_id] = []
    for captions_file in [captions_file_1, captions_file_2]:
        with open(captions_file) as file:
            for line in file:
                splitted = line.split("\t")
                if len(splitted) > 1:
                    # print(line)
                    image_id = splitted[0]
                    caption_id = splitted[1]
                    caption = splitted[2]

                    # remove special chars, make caption lower case
                    caption = caption.replace("\n", "").replace('"', "").lower()
                    caption = caption.translate(
                        str.maketrans(dict.fromkeys(string.punctuation))
                    )

                    # Tokenize the caption
                    caption = nltk.word_tokenize(caption)

                    # Cut off too long captions
                    caption = caption[:MAX_CAPTION_LEN]

                    word_freq.update(caption)

                    captions[int(image_id)].append(caption)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Print the most frequent words
    print(f"Most frequent words: {word_freq.most_common(1000)}")

    # Create vocab
    vocab = Vocab(
        word_freq,
        specials=[TOKEN_PADDING, Vocab.UNK, TOKEN_START, TOKEN_END],
        min_freq=vocab_min_freq,
    )
    vocab_path = os.path.join(output_folder, VOCAB_FILENAME)

    print(f"Vocab size: {len(vocab)}")
    print("Saving new vocab to {}".format(vocab_path))
    with open(vocab_path, "wb") as file:
        pickle.dump(vocab, file)

    # Discard images with not enough captions
    images = {id: image for id, image in enumerate(images)}
    images = {
        id: image
        for id, image in images.items()
        if len(captions[id]) == CAPTIONS_PER_IMAGE
    }
    captions = {
        id: captions_image
        for id, captions_image in captions.items()
        if id in images.keys()
    }

    # Encode the captions using the vocab
    captions = {
        id: encode_captions(captions_image, vocab)
        for id, captions_image in captions.items()
    }

    # Create dataset splits
    all_indices = list(images.keys())
    indices = {}
    indices["train"], indices["test"] = train_test_split(
        all_indices, test_size=0.1, random_state=RANDOM_SEED
    )
    indices["train"], indices["val"] = train_test_split(
        indices["train"], test_size=VAL_SET_SIZE, random_state=RANDOM_SEED
    )

    for split in ["train", "val", "test"]:
        images_split = {i: images[i] for i in indices[split]}
        captions_split = {i: captions[i] for i in indices[split]}

        # Create hdf5 file and dataset for the images
        images_dataset_path = os.path.join(output_folder, IMAGES_FILENAME[split])
        print("Creating image dataset at {}".format(images_dataset_path))
        with h5py.File(images_dataset_path, "a") as h5py_file:
            for img_id, img in tqdm(images_split.items()):
                # Read image and save it to hdf5 file
                h5py_file.create_dataset(
                    str(img_id), (224, 224, 3), dtype="uint8", data=img
                )

        # Save captions
        captions_path = os.path.join(output_folder, CAPTIONS_FILENAME[split])
        print("Saving captions to {}".format(captions_path))
        with open(captions_path, "wb") as file:
            pickle.dump(captions_split, file)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        help="Folder where the abstract scenes dataset is located",
        default=os.path.expanduser("~/data/abstract_scenes/AbstractScenes_v1.1/"),
    )
    parser.add_argument(
        "--output-folder",
        help="Folder in which the preprocessed data should be stored",
        default=DATA_PATH,
    )
    parser.add_argument(
        "--vocab-min-freq",
        help="Minimum number of occurrences for a word to be included in the vocabulary.",
        type=int,
        default=5,
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    preprocess_images_and_captions(
        parsed_args.dataset_folder,
        parsed_args.output_folder,
        parsed_args.vocab_min_freq,
    )
