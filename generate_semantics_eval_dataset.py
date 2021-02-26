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

# OBJECTS_ANIMALS = ["dog", "bear", "cat", "snake", "owl", "duck"]
OBJECTS_ANIMALS = ["dog", "cat"]
OBJECTS_INANIMATE = ["pie", "pizza"]


VERBS_INTRANSITIVE = ["sitting", "standing", "running"]
VERBS_TRANSITIVE = ["eating", "waving", "throwing", "kicking"]

VOCAB_TO_OBJECT_NAMES = {
    "mike": ["Boy"],
    "jenny": ["Girl"],
    "dog": ["Dog"],
    "bear": ["Bear"],
    "cat": ["Cat"],
    "snake": ["Snake"],
    "owl": ["Owl"],
    "duck": ["Duck"],
    "ball": ["Baseball", "BeachBall", "Basketball", "SoccerBall", "TennisBall", "Football"],
    "hat": ["ChefHat", "PirateHat", "WizardHat", "VikingHat", "BaseballCap", "WinterCap", "Bennie"],
    "tree": ["PineTree", "OakTree", "AppleTree"],
    "table": ["Table"],
    "sandbox": ["Sandbox"],
    "slide": ["Slide"],
    "hamburger": ["Hamburger"],
    "pizza": ["Pizza"],
    "pie": ["Pie"],
}


def contains_instance(meta_data, img_id, instance_name):
    meta = meta_data[img_id]

    object_names = VOCAB_TO_OBJECT_NAMES[instance_name]
    for object_name in object_names:

        index = META_DATA_DICT[object_name].values[0]
        if int(meta[index]) == 1:
            return True

    return False


def get_image_ids_single_actor(image_ids, meta_data):
    ids = []
    for img_id in image_ids:
        if (contains_instance(meta_data, img_id, "mike") and not contains_instance(meta_data, img_id, "jenny")) or (
                contains_instance(meta_data, img_id, "jenny") and not contains_instance(meta_data, img_id, "mike")):
            ids.append(img_id)

    return ids


def get_image_ids_one_object(image_ids, meta_data, objects):
    """Return all images that contain exactly one of the objects in the given list."""
    ids = []
    for img_id in image_ids:
        instance_counter = 0
        for object in objects:
            if contains_instance(meta_data, img_id, object):
                instance_counter += 1
            if instance_counter > 1:
                break
        if instance_counter == 1:
            ids.append(img_id)

    return ids


def generate_eval_set_persons(image_ids, meta_data, images, captions, vocab):
    samples = []

    image_ids = get_image_ids_single_actor(image_ids, meta_data)
    for img_id in image_ids:
        for target_caption in captions[img_id]:
            target_caption = decode_caption(target_caption, vocab, join=False)
            for actor in ["jenny", "mike"]:
                if actor in target_caption:
                    if actor == "jenny":
                        for img_id_distractor in image_ids:
                            for distractor_caption in captions[img_id_distractor]:
                                distractor_caption = decode_caption(distractor_caption, vocab, join=False)
                                if "mike" in distractor_caption:
                                    replaced = [word if word != "mike" else "jenny" for word in distractor_caption]
                                    if replaced == target_caption:
                                        print(target_caption)
                                        print(distractor_caption)
                                        target_sentence = " ".join(target_caption)
                                        distractor_sentence = " ".join(distractor_caption)
                                        samples.append({"img_id": img_id, "target_sentence": target_sentence,
                                                        "distractor_sentence": distractor_sentence})
                                        samples.append({"img_id": img_id_distractor, "target_sentence": distractor_sentence,
                                                        "distractor_sentence": target_sentence})

                            # else:
                    #     for img_id_distractor in image_ids:
                    #         for caption_distractor in captions[img_id_distractor]:
                    #             distractor = decode_caption(caption_distractor, vocab, join=False)
                    #             replaced = [word if word != "jenny" else "mike" for word in distractor]
                    #             if replaced == target_caption:
                    #                 print(target_caption)

                    # for object_distractor in ["jenny", "mike"]:
                    #     if actor != object_distractor:
                    #         distractor = [word if word != actor else object_distractor for word in decoded_caption]
                    #         if actor == "jenny":
                    #             distractor = [word if word != "her" else "his" for word in distractor]
                    #         else:
                    #             distractor = [word if word != "his" else "her" for word in distractor]

                            # target_sentence = " ".join(target_caption)
                            # distractor_sentence = " ".join(distractor_caption)
                            # samples.append({"img_id": img_id, "target_sentence": target_sentence, "distractor_sentence": distractor_sentence})
                            # print(f"{img_id},{target_sentence},{distractor_sentence}")

        # show_image(images[str(img_id)])
    data = pd.DataFrame(samples)
    return data


def generate_eval_set_agent_patient(image_ids, meta_data, images, captions, vocab):
    samples = []

    for img_id in image_ids:
        for caption in captions[img_id]:
            decoded_caption = decode_caption(caption, vocab, join=False)
            if "jenny" in decoded_caption and "mike" in decoded_caption:
                if not "and" in decoded_caption:
                    replacements = {"jenny": "mike", "mike": "jenny"}
                    distractor = [replacements.get(word, word) for word in decoded_caption]

                    target_sentence = " ".join(decoded_caption)
                    distractor_sentence = " ".join(distractor)
                    samples.append({"img_id": img_id, "target_sentence": target_sentence, "distractor_sentence": distractor_sentence})
                    # print(f"{img_id},{target_sentence},{distractor_sentence}")

                    # show_image(images[str(img_id)])
    data = pd.DataFrame(samples)
    return data


def generate_eval_set_objects(image_ids, meta_data, images, captions, vocab, objects):
    samples = []

    image_ids = get_image_ids_one_object(image_ids, meta_data, objects)
    for img_id in image_ids:
        for caption in captions[img_id]:
            decoded_caption = decode_caption(caption, vocab, join=False)
            for obj in objects:
                if obj in decoded_caption:
                    for object_distractor in objects:
                        if obj != object_distractor:
                            # Create one distractor for each other object
                            distractor = [word if word != obj else object_distractor for word in decoded_caption]

                            target_sentence = " ".join(decoded_caption)
                            distractor_sentence = " ".join(distractor)
                            samples.append({"img_id": img_id, "target_sentence": target_sentence, "distractor_sentence": distractor_sentence})
                            # print(f"{img_id},{target_sentence},{distractor_sentence}")

        # show_image(images[str(img_id)])
    data = pd.DataFrame(samples)
    return data


def generate_eval_set_verbs(image_ids, meta_data, images, captions, vocab, verbs):
    samples = []

    image_ids = get_image_ids_single_actor(image_ids, meta_data)
    for img_id in image_ids:
        for caption in captions[img_id]:
            decoded_caption = decode_caption(caption, vocab, join=False)
            for verb in verbs:
                if verb in decoded_caption:
                    # Cut off sentence after verb
                    decoded_caption = decoded_caption[:decoded_caption.index(verb)+1]

                    if ("jenny" in decoded_caption) or ("mike" in decoded_caption):

                        for verb_distractor in verbs:
                            if verb != verb_distractor:
                                # Create one distractor for each other object
                                distractor = [word if word != verb else verb_distractor for word in decoded_caption]

                                target_sentence = " ".join(decoded_caption)
                                distractor_sentence = " ".join(distractor)
                                samples.append({"img_id": img_id, "target_sentence": target_sentence, "distractor_sentence": distractor_sentence})
                                # print(f"{img_id},{target_sentence},{distractor_sentence}")

        # show_image(images[str(img_id)])
    data = pd.DataFrame(samples)
    return data


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

    data_persons = generate_eval_set_persons(image_ids.copy(), meta_data, images, captions, vocab)
    data_persons.to_csv("data/semantics_eval_persons.csv", index=False)

    # data_animals = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, OBJECTS_ANIMALS)
    # data_animals.to_csv("data/semantics_eval_animals.csv", index=False)
    #
    # data_inanimates = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, OBJECTS_INANIMATE)
    # data_inanimates.to_csv("data/semantics_eval_inanimates.csv", index=False)
    #
    # data_verbs_1 = generate_eval_set_verbs(image_ids.copy(), meta_data, images, captions, vocab, VERBS_INTRANSITIVE)
    # data_verbs_2 = generate_eval_set_verbs(image_ids.copy(), meta_data, images, captions, vocab, VERBS_TRANSITIVE)
    # pd.concat((data_verbs_1, data_verbs_2)).to_csv("data/semantics_eval_verbs_intransitive.csv", index=False)
    #
    # data_agent_patient = generate_eval_set_agent_patient(image_ids.copy(), meta_data, images, captions, vocab)
    # data_agent_patient.to_csv("data/semantics_eval_agent_patient.csv", index=False)


def get_args():
    parser = argparse.ArgumentParser()

    return core.init(parser)


if __name__ == "__main__":
    args = get_args()
    main(args)
