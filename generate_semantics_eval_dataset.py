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
OBJECTS_ANIMALS = ["dog", "cat"] #, "bear", "snake", "owl", "duck"]
OBJECTS_ANIMALS_2 = ["snake", "duck"]
OBJECTS_INANIMATE = ["pie", "pizza"]
OBJECTS_INANIMATE_2 = ["ball", "frisbee"] # ["basketball", "football"] # table tree

VERBS_1 = ["sitting", "standing"] #, "running"]
VERBS_2 = ["sitting", "running"] #["eating", "kicking"] #, "throwing", "kicking"]
VERBS_3 = ["eating", "playing"] #["eating", "kicking"] #, "throwing", "kicking"]

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
    "basketball": ["Basketball"],
    "football": ["Football"],
    "frisbee": ["Frisbee"],
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
            actor = "jenny"
            distractor = "mike"
            if actor in target_caption:
                for img_id_distractor in image_ids:
                    for distractor_caption in captions[img_id_distractor]:
                        distractor_caption = decode_caption(distractor_caption, vocab, join=False)
                        if distractor in distractor_caption:
                            replaced = [word if word != distractor else actor for word in distractor_caption]
                            if replaced == target_caption:
                                # print(target_caption)
                                # print(distractor_caption)
                                target_sentence = " ".join(target_caption)
                                distractor_sentence = " ".join(distractor_caption)
                                sample_1 = {"img_id": img_id, "target_sentence": target_sentence,
                                                "distractor_sentence": distractor_sentence}
                                sample_2 = {"img_id": img_id_distractor, "target_sentence": distractor_sentence,
                                                "distractor_sentence": target_sentence}
                                if sample_1 not in samples and sample_2 not in samples:
                                    samples.append(sample_1)
                                    samples.append(sample_2)
                                # show_image(images[str(img_id)])
    data = pd.DataFrame(samples)
    data = data.drop_duplicates()
    return data


def generate_eval_set_agent_patient(image_ids, meta_data, images, captions, vocab):
    samples = []

    for img_id in image_ids:
        for target_caption in captions[img_id]:
            target_caption = decode_caption(target_caption, vocab, join=False)
            actor = "jenny"
            distractor = "mike"
            if actor in target_caption and distractor in target_caption:
                # Cut off sentence after object
                if target_caption.index(actor) < target_caption.index(distractor):
                    target_caption = target_caption[:target_caption.index(distractor) + 1]
                else:
                    target_caption = target_caption[:target_caption.index(actor) + 1]

                if "and" not in target_caption:
                    for img_id_distractor in image_ids:
                        for distractor_caption in captions[img_id_distractor]:
                            distractor_caption = decode_caption(distractor_caption, vocab, join=False)

                            replacements = {"jenny": "mike", "mike": "jenny"}
                            replaced = [replacements.get(word, word) for word in distractor_caption]

                            if replaced == target_caption:
                                print(target_caption)
                                # print(distractor_caption)
                                target_sentence = " ".join(target_caption)
                                distractor_sentence = " ".join(distractor_caption)
                                sample_1 = {"img_id": img_id, "target_sentence": target_sentence,
                                            "distractor_sentence": distractor_sentence}
                                sample_2 = {"img_id": img_id_distractor, "target_sentence": distractor_sentence,
                                            "distractor_sentence": target_sentence}
                                if sample_1 not in samples and sample_2 not in samples:
                                    samples.append(sample_1)
                                    samples.append(sample_2)
                                    print(img_id)
                                    show_image(images[str(img_id)])

    data = pd.DataFrame(samples)
    data = data.drop_duplicates()
    return data


def generate_eval_set_objects(image_ids, meta_data, images, captions, vocab, objects):
    samples = []

    image_ids = get_image_ids_one_object(image_ids, meta_data, objects)
    for img_id in image_ids:
        for target_caption in captions[img_id]:
            target_caption = decode_caption(target_caption, vocab, join=False)
            for target in objects:
                if target in target_caption:
                    for img_id_distractor in image_ids:
                        for distractor_caption in captions[img_id_distractor]:
                            distractor_caption = decode_caption(distractor_caption, vocab, join=False)
                            for distractor in objects:
                                if distractor != target:
                                    if distractor in distractor_caption:
                                        replaced = [word if word != distractor else target for word in
                                                    distractor_caption]
                                        if replaced == target_caption:
                                            # print(target_caption)
                                            # print(distractor_caption)
                                            target_sentence = " ".join(target_caption)
                                            distractor_sentence = " ".join(distractor_caption)
                                            sample_1 = {"img_id": img_id, "target_sentence": target_sentence,
                                                        "distractor_sentence": distractor_sentence}
                                            sample_2 = {"img_id": img_id_distractor,
                                                        "target_sentence": distractor_sentence,
                                                        "distractor_sentence": target_sentence}
                                            if sample_1 not in samples and sample_2 not in samples:
                                                samples.append(sample_1)
                                                samples.append(sample_2)

        # show_image(images[str(img_id)])
    data = pd.DataFrame(samples)
    data = data.drop_duplicates()

    return data


def generate_eval_set_verbs(image_ids, meta_data, images, captions, vocab, verbs):
    samples = []

    image_ids = get_image_ids_single_actor(image_ids, meta_data)

    for img_id in image_ids:
        for target_caption in captions[img_id]:
            target_caption = decode_caption(target_caption, vocab, join=False)
            for target in verbs:
                if target in target_caption:
                    # Cut off sentence after verb
                    target_caption = target_caption[:target_caption.index(target) + 1]
                    if ("jenny" in target_caption) or ("mike" in target_caption):
                        for img_id_distractor in image_ids:
                            for distractor_caption in captions[img_id_distractor]:
                                distractor_caption = decode_caption(distractor_caption, vocab, join=False)
                                for distractor in verbs:
                                    if distractor != target:
                                        if distractor in distractor_caption:
                                            # Cut off sentence after verb
                                            distractor_caption = distractor_caption[:distractor_caption.index(distractor) + 1]
                                            replaced = [word if word != distractor else target for word in
                                                        distractor_caption]
                                            if replaced == target_caption:
                                                print(target_caption)
                                                print(distractor_caption)
                                                target_sentence = " ".join(target_caption)
                                                distractor_sentence = " ".join(distractor_caption)
                                                sample_1 = {"img_id": img_id, "target_sentence": target_sentence,
                                                            "distractor_sentence": distractor_sentence}
                                                sample_2 = {"img_id": img_id_distractor,
                                                            "target_sentence": distractor_sentence,
                                                            "distractor_sentence": target_sentence}
                                                if sample_1 not in samples and sample_2 not in samples:
                                                    samples.append(sample_1)
                                                    samples.append(sample_2)

    data = pd.DataFrame(samples)
    data = data.drop_duplicates()
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

    data_animals_1 = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, OBJECTS_ANIMALS)
    data_animals_2 = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, OBJECTS_ANIMALS_2)
    pd.concat((data_animals_1, data_animals_2)).to_csv("data/semantics_eval_animals.csv", index=False)

    data_inanimates_1 = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, OBJECTS_INANIMATE)
    data_inanimates_2 = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, OBJECTS_INANIMATE_2)
    pd.concat((data_inanimates_1, data_inanimates_2)).to_csv("data/semantics_eval_inanimates.csv", index=False)

    data_verbs_1 = generate_eval_set_verbs(image_ids.copy(), meta_data, images, captions, vocab, VERBS_1)
    data_verbs_2 = generate_eval_set_verbs(image_ids.copy(), meta_data, images, captions, vocab, VERBS_2)
    data_verbs_3 = generate_eval_set_verbs(image_ids.copy(), meta_data, images, captions, vocab, VERBS_3)
    pd.concat((data_verbs_1, data_verbs_2, data_verbs_3)).to_csv("data/semantics_eval_verbs.csv", index=False)

    data_agent_patient = generate_eval_set_agent_patient(image_ids.copy(), meta_data, images, captions, vocab)
    data_agent_patient.to_csv("data/semantics_eval_agent_patient.csv", index=False)


def get_args():
    parser = argparse.ArgumentParser()

    return core.init(parser)


if __name__ == "__main__":
    args = get_args()
    main(args)
