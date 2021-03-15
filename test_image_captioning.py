from __future__ import print_function

import argparse
import pickle
import os

import h5py
import numpy as np

import torch
import torch.distributions
import torch.utils.data

from dataset import SemanticsEvalDataset, CaptionDataset
from models.image_captioning.show_and_tell import ShowAndTell
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker, cosine_sim
from models.image_sentence_ranking.ranking_model_grounded import ImageSentenceRankerGrounded
from models.joint.joint_learner import JointLearner
from models.joint.joint_learner_sat import JointLearnerSAT
from models.language_modeling.language_model import LanguageModel
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH, show_image,
)
from train_image_captioning import print_captions
from utils import decode_caption, CHECKPOINT_DIR_IMAGE_CAPTIONING, SEMANTICS_EVAL_FILES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_test_images_captions(model, dataloader, images_data, vocab, args):
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, target_captions, caption_lengths, image_ids) in enumerate(dataloader):
            captions, _, _ = model.decode_nucleus_sampling(images, 1, top_p=0.9)

            print_captions(captions, target_captions, image_ids, vocab)

            if args.show_image:
                image_data = images_data[str(image_ids[0].item())][()]
                image = torch.FloatTensor(image_data)
                features_scale_factor = 255
                image = image / features_scale_factor
                show_image(image)


def main(args):
    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    print("Loading model checkpoint from {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "show_attend_and_tell" in args.checkpoint:
        print("Loading sat image captioning model.")
        word_embedding_size = 128
        lstm_hidden_size = 512
        model = ShowAttendAndTell(word_embedding_size, lstm_hidden_size, vocab, MAX_CAPTION_LEN,
                                  fine_tune_resnet=False)

    elif "show_and_tell" in args.checkpoint:
        print("Loading st image captioning model.")
        word_embedding_size = 512
        visual_embedding_size = 512
        lstm_hidden_size = 512
        model = ShowAndTell(
            word_embedding_size,
            visual_embedding_size,
            lstm_hidden_size,
            vocab,
            MAX_CAPTION_LEN,
            fine_tune_resnet=False,
        )

    elif "joint" in args.checkpoint:
        print('Loading joint learner model.')
        word_embedding_size = 512
        joint_embeddings_size = 512
        lstm_hidden_size = 512
        model = JointLearnerSAT(
            word_embedding_size,
            lstm_hidden_size,
            vocab,
            MAX_CAPTION_LEN,
            joint_embeddings_size,
            fine_tune_resnet=False,
        )

    else:
        raise RuntimeError(f"Non supported model: {args.checkpoint}")

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    test_images_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_PATH, IMAGES_FILENAME["test"], CAPTIONS_FILENAME["test"], vocab),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )

    images = h5py.File(
        os.path.join(DATA_PATH, IMAGES_FILENAME["test"]), "r"
    )


    print_test_images_captions(model, test_images_loader, images, vocab, args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
    )
    parser.add_argument(
        "--show-image",
        default=False,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Start test on device: ", device)
    args = get_args()
    main(args)
