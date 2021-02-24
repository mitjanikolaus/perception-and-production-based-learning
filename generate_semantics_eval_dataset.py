from __future__ import print_function

import argparse
import pickle
import os

import h5py
import pandas as pd

import egg.core as core
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    DATA_PATH, show_image,
)
from utils import decode_caption

META_DATA_PATH = os.path.expanduser("~/data/abstract_scenes/AbstractScenes_v1.1/VisualFeatures/10K_instance_occurence_58.txt")
META_DATA_DICT_PATH = os.path.expanduser("~/data/abstract_scenes/AbstractScenes_v1.1/VisualFeatures/10K_instance_occurence_58_names.txt")

META_DATA_DICT = pd.read_csv(META_DATA_DICT_PATH, sep="\t", index_col=0, names=["id"]).T


def contains_instance(meta_data, img_id, instance_name):
    meta = meta_data[img_id]

    index = META_DATA_DICT[instance_name].values[0]

    return int(meta[index]) == 1


def get_image_ids_single_actor(image_ids, meta_data):
    ids = []
    for img_id in image_ids:
        if (contains_instance(meta_data, img_id, "Boy") and not contains_instance(meta_data, img_id, "Girl")) or (
                contains_instance(meta_data, img_id, "Girl") and not contains_instance(meta_data, img_id, "Boy")):
            ids.append(img_id)

    return ids


def generate_persons_set(image_ids, meta_data, images, captions, vocab):
    samples = []

    image_ids = get_image_ids_single_actor(image_ids, meta_data)
    for img_id in image_ids:
        for caption in captions[img_id]:
            decoded_caption = decode_caption(caption, vocab)
            if "jenny" in decoded_caption or "mike" in decoded_caption:
                distractor = decoded_caption.replace("jenny", "XXXX").replace("mike", "jenny").replace("XXXX", "mike")
                distractor = distractor.replace(" his ", "XXXX").replace(" her ", " his ").replace("XXXX", " her ")
                samples.append({"img_id": img_id, "target_sentence": decoded_caption, "distractor_sentence": distractor})
                # print(f"{img_id},{decoded_caption},{distractor}")

        # show_image(images[str(img_id)])
    data = pd.DataFrame(samples)
    data.to_csv("data/semantics_eval_persons.csv", index=False)


def main(args):
    meta_data = {}
    with open(META_DATA_PATH) as file:
        for img_id, line in enumerate(file):
            splitted = line.split("\t")
            meta_data[img_id] = splitted

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

    image_ids = [int(key) for key in images.keys()]
    generate_persons_set(image_ids.copy(), meta_data, images, captions, vocab)



def get_args():
    parser = argparse.ArgumentParser()

    return core.init(parser)


if __name__ == "__main__":
    args = get_args()
    main(args)
