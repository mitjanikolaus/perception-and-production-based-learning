import argparse
import itertools
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

META_DATA_ADJECTIVES_PATH = os.path.expanduser("~/data/abstract_scenes/AbstractScenes_v1.1/VisualFeatures/10K_person_24.txt")
# This is a fixed version (the original one from the dataset is not correct)
META_DATA_ADJECTIVES_DICT_PATH = os.path.expanduser("data/10K_person_24_names.txt")
META_DATA_ADJECTIVES_DICT = pd.read_csv(META_DATA_ADJECTIVES_DICT_PATH, sep="\t", index_col=0, names=["id"]).T

OBJECTS_ANIMALS = ["dog", "cat", "snake", "bear", "duck", "owl"]
TUPLES_ANIMALS = list(itertools.combinations(OBJECTS_ANIMALS, 2))

OBJECTS_INANIMATE = ["ball", "hat", "tree", "table", "sandbox", "slide", "sunglasses", "pie", "pizza", "hamburger", "balloons", "frisbee"]
TUPLES_INANIMATE = list(itertools.combinations(OBJECTS_INANIMATE, 2))

VERBS = [("sitting", "standing"), ("sitting", "running"), ("eating", "playing"), ("eating", "kicking"), ("throwing", "eating"), ("throwing", "kicking"), ("sitting", "kicking"), ("jumping", "sitting")]

ADJECTIVES = [("happy", "sad"), ("happy", "angry"), ("happy", "upset"), ("happy", "scared"), ("happy", "mad"), ("happy", "afraid"), ("happy", "surprised")]

ADJECTIVES_NEGATIVE = ["angry", "mad", "sad", "upset", "scared", "afraid", "surprised"]

VOCAB_TO_FACE_EXPRESSION = {
    "angry": ["Expression_Angry"],
    "mad": ["Expression_Angry"],
    "sad": ["Expression_Sad"],
    "upset": ["Expression_Surprise"],
    "scared": ["Expression_Surprise"],
    "afraid": ["Expression_Surprise"],
    "surprised": ["Expression_Surprise"],
    "happy": ["Expression_Smile", "Expression_Laugh"],
}

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
    "sunglasses": ["Sunglasses"],
    "balloons": ["Balloons"],
}


def contains_actor_with_attribute(meta_data_adjectives, img_id, actor, attribute):
    meta = meta_data_adjectives[img_id]
    attribute_names = VOCAB_TO_FACE_EXPRESSION[attribute]
    for attribute_name in attribute_names:
        if actor == "jenny":
            attribute_name = "Girl_"+attribute_name
        else:
            attribute_name = "Boy_"+attribute_name
        index = META_DATA_ADJECTIVES_DICT[attribute_name].values[0]
        if int(meta[index]) == 1:
            return True
    return False


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


def get_image_ids_two_actors(image_ids, meta_data):
    ids = []
    for img_id in image_ids:
        if (contains_instance(meta_data, img_id, "mike") and contains_instance(meta_data, img_id, "jenny")):
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


def find_minimal_pairs(image_ids, meta_data, images, captions, vocab):
    samples = []

    for img_id in image_ids:
        for target_caption in captions[img_id]:
            target_caption = decode_caption(target_caption, vocab, join=False)

            for img_id_distractor in image_ids:
                for distractor_caption in captions[img_id_distractor]:
                    distractor_caption = decode_caption(distractor_caption, vocab, join=False)

                    if target_caption != distractor_caption:
                        for word_1 in target_caption:
                            for word_2 in target_caption:
                                if word_1 != word_2:
                                    permuted = []
                                    for word in target_caption:
                                        if word == word_1:
                                            permuted.append(word_2)
                                        elif word == word_2:
                                            permuted.append(word_1)
                                        else:
                                            permuted.append(word)
                                    if permuted == distractor_caption:
                                        if word_1 != "jenny" and word_2 != "jenny":
                                            print(target_caption)
                                            print(permuted)
    return samples


def generate_eval_set_persons(image_ids, meta_data, images, captions, vocab):
    samples = []

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


def generate_eval_set_objects(image_ids, meta_data, images, captions, vocab, target_tuples):
    samples = []

    for target_tuple in target_tuples:
        image_ids_one_object = get_image_ids_one_object(image_ids, meta_data, target_tuple)
        for img_id in image_ids_one_object:
            for target_caption in captions[img_id]:
                target_caption = decode_caption(target_caption, vocab, join=False)
                for target in target_tuple:
                    if target in target_caption:
                        for img_id_distractor in image_ids_one_object:
                            for distractor_caption in captions[img_id_distractor]:
                                distractor_caption = decode_caption(distractor_caption, vocab, join=False)
                                for distractor in target_tuple:
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

    return data


def generate_eval_set_adjectives_hard(image_ids, meta_data_adjectives, images, captions, vocab, target_tuples):
    samples = []

    for target_tuple in target_tuples:
        for img_id in image_ids:
            for target_caption in captions[img_id]:
                target_caption = decode_caption(target_caption, vocab, join=False)
                for target in target_tuple:
                    if target in target_caption:
                        # Cut off sentence after verb/adjective
                        target_caption = target_caption[:target_caption.index(target) + 1]
                        if ("jenny" in target_caption) or ("mike" in target_caption):
                            for img_id_distractor in image_ids:
                                for distractor_caption in captions[img_id_distractor]:
                                    distractor_caption = decode_caption(distractor_caption, vocab, join=False)
                                    for distractor in target_tuple:
                                        if distractor != target:
                                            if distractor in distractor_caption:
                                                # Cut off sentence after verb/adjective
                                                distractor_caption = distractor_caption[:distractor_caption.index(distractor) + 1]
                                                replaced = [word if word != distractor else target for word in
                                                            distractor_caption]
                                                if replaced == target_caption:
                                                    # filter for cases where other actor has different mood
                                                    if "jenny" in target_caption:
                                                        actor_target = "jenny"
                                                        actor_distractor = "mike"
                                                    else:
                                                        actor_target = "mike"
                                                        actor_distractor = "jenny"
                                                    # show_image(images[str(img_id)])
                                                    # show_image(images[str(img_id_distractor)])
                                                    if target == "happy":
                                                        if contains_actor_with_attribute(meta_data_adjectives, img_id, actor_distractor, target):
                                                            continue
                                                        contains_actor = False
                                                        for adjective in ADJECTIVES_NEGATIVE:
                                                            if contains_actor_with_attribute(meta_data_adjectives,
                                                                                             img_id_distractor, actor_distractor,
                                                                                             adjective):
                                                                contains_actor = True
                                                        if contains_actor:
                                                            continue
                                                        if not contains_actor_with_attribute(meta_data_adjectives,
                                                                                             img_id, actor_target,
                                                                                             target):
                                                            continue

                                                        contains_actor = False
                                                        for adjective in ADJECTIVES_NEGATIVE:
                                                            if contains_actor_with_attribute(meta_data_adjectives,
                                                                                                 img_id_distractor, actor_target,
                                                                                                 adjective):
                                                                contains_actor = True
                                                        if not contains_actor:
                                                            continue
                                                    else:
                                                        contains_actor = False
                                                        for adjective in ADJECTIVES_NEGATIVE:
                                                            if contains_actor_with_attribute(meta_data_adjectives, img_id, actor_distractor, adjective):
                                                                contains_actor = True
                                                        if contains_actor:
                                                            continue
                                                        if contains_actor_with_attribute(meta_data_adjectives, img_id_distractor, actor_distractor, distractor):
                                                            continue

                                                        contains_actor = False
                                                        for adjective in ADJECTIVES_NEGATIVE:
                                                            if contains_actor_with_attribute(meta_data_adjectives,
                                                                                                 img_id, actor_target,
                                                                                                 adjective):
                                                                contains_actor = True
                                                        if not contains_actor:
                                                            continue

                                                        if not contains_actor_with_attribute(meta_data_adjectives,
                                                                                             img_id_distractor,
                                                                                             actor_target, distractor):
                                                            continue

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
                                                        # print(target_caption)
                                                        # print(distractor_caption)
                                                        # show_image(images[str(img_id)])
                                                        # show_image(images[str(img_id_distractor)])

    data = pd.DataFrame(samples)
    return data


def generate_eval_set_verbs_or_adjectives(image_ids, meta_data, images, captions, vocab, target_tuples):
    samples = []

    for target_tuple in target_tuples:
        for img_id in image_ids:
            for target_caption in captions[img_id]:
                target_caption = decode_caption(target_caption, vocab, join=False)
                for target in target_tuple:
                    if target in target_caption:
                        # Cut off sentence after verb/adjective
                        target_caption = target_caption[:target_caption.index(target) + 1]
                        if ("jenny" in target_caption) or ("mike" in target_caption):
                            for img_id_distractor in image_ids:
                                for distractor_caption in captions[img_id_distractor]:
                                    distractor_caption = decode_caption(distractor_caption, vocab, join=False)
                                    for distractor in target_tuple:
                                        if distractor != target:
                                            if distractor in distractor_caption:
                                                # Cut off sentence after verb/adjective
                                                distractor_caption = distractor_caption[:distractor_caption.index(distractor) + 1]
                                                replaced = [word if word != distractor else target for word in
                                                            distractor_caption]
                                                if replaced == target_caption:
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
                                                        print(img_id)
                                                        print(img_id_distractor)
                                                        print(target_caption)
                                                        print(distractor_caption)
                                                        show_image(images[str(img_id)])
                                                        show_image(images[str(img_id_distractor)])

    data = pd.DataFrame(samples)
    return data


def main(args):
    meta_data = {}
    with open(META_DATA_PATH) as file:
        for img_id, line in enumerate(file):
            splitted = line.split("\t")
            meta_data[img_id] = splitted

    meta_data_adjectives = {}
    with open(META_DATA_ADJECTIVES_PATH) as file:
        for img_id, line in enumerate(file):
            splitted = line.split("\t")
            meta_data_adjectives[img_id] = splitted

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
    image_ids_single_actor = get_image_ids_single_actor(image_ids, meta_data)
    image_ids_two_actors = get_image_ids_two_actors(image_ids, meta_data)

    # data_persons = generate_eval_set_persons(image_ids_single_actor.copy(), meta_data, images, captions, vocab)
    # data_persons.to_csv("data/semantics_eval_persons.csv", index=False)

    # data_animals = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, TUPLES_ANIMALS)
    # data_animals.to_csv("data/semantics_eval_animals.csv", index=False)

    # data_inanimates = generate_eval_set_objects(image_ids.copy(), meta_data, images, captions, vocab, TUPLES_INANIMATE)
    # data_inanimates.to_csv("data/semantics_eval_inanimates.csv", index=False)

    # data_verbs = generate_eval_set_verbs_or_adjectives(image_ids_single_actor.copy(), meta_data, images, captions,
    #                                                    vocab, VERBS)
    # data_verbs.to_csv("data/semantics_eval_verbs.csv", index=False)

    data_verbs = generate_eval_set_verbs_or_adjectives(image_ids_two_actors.copy(), meta_data, images, captions,
                                                       vocab, VERBS)
    data_verbs.to_csv("data/semantics_eval_verb_noun_binding.csv", index=False)

    # data_adj = generate_eval_set_verbs_or_adjectives(image_ids_single_actor.copy(), meta_data, images, captions, vocab, ADJECTIVES)
    # data_adj.to_csv("data/semantics_eval_adjectives.csv", index=False)
    #
    # data_adj = generate_eval_set_adjectives_hard(image_ids_two_actors.copy(), meta_data_adjectives, images, captions, vocab,
    #                                                  ADJECTIVES)
    # data_adj.to_csv("data/semantics_eval_adjective_noun_binding.csv", index=False)
    #
    # data_agent_patient = generate_eval_set_agent_patient(image_ids.copy(), meta_data, images, captions, vocab)
    # data_agent_patient.to_csv("data/semantics_eval_semantic_roles.csv", index=False)


def get_args():
    parser = argparse.ArgumentParser()

    return core.init(parser)


if __name__ == "__main__":
    args = get_args()
    main(args)
