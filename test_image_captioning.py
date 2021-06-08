from __future__ import print_function

import argparse
import pickle
import os

import torch
import torch.distributions
import torch.utils.data
from torch.utils.data import DataLoader

from dataset import CaptionRLDataset
from eval_semantics import get_semantics_eval_dataloader
from models.image_captioning.show_and_tell import ShowAndTell
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.joint.joint_learner import JointLearner
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH,
)
from train_image_captioning import validate_model
from utils import DEFAULT_WORD_EMBEDDINGS_SIZE, DEFAULT_LSTM_HIDDEN_SIZE, SEMANTICS_EVAL_FILES, set_seeds

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    if args.seed:
        set_seeds(args.seed)

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    test_loader = DataLoader(
        CaptionRLDataset(
            DATA_PATH, IMAGES_FILENAME["test"], CAPTIONS_FILENAME["test"], vocab
        ),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionRLDataset.pad_collate,
    )
    semantics_eval_loaders = {
        file: get_semantics_eval_dataloader(file, vocab)
        for file in SEMANTICS_EVAL_FILES
    }

    bleu_scores = []

    for root, dirs, files in os.walk(args.checkpoints_dir):
        for name in files:
            checkpoint_path = os.path.join(root, name)
            if checkpoint_path.endswith(".pt"):
                print("Loading model checkpoint from {}".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path, map_location=device)

                word_embedding_size = DEFAULT_WORD_EMBEDDINGS_SIZE
                visual_embedding_size = DEFAULT_LSTM_HIDDEN_SIZE
                joint_embeddings_size = visual_embedding_size
                lstm_hidden_size = DEFAULT_LSTM_HIDDEN_SIZE

                if "show_attend_and_tell" in checkpoint_path:
                    print("Loading sat image captioning model.")
                    args.model = "show_attend_and_tell"
                    model = ShowAttendAndTell(
                        word_embedding_size,
                        lstm_hidden_size,
                        vocab,
                        MAX_CAPTION_LEN,
                        fine_tune_resnet=False,
                    )

                elif "show_and_tell" in checkpoint_path:
                    print("Loading st image captioning model.")
                    args.model = "show_and_tell"
                    model = ShowAndTell(
                        word_embedding_size,
                        visual_embedding_size,
                        lstm_hidden_size,
                        vocab,
                        MAX_CAPTION_LEN,
                        fine_tune_resnet=False,
                    )

                elif "joint" in checkpoint_path:
                    print("Loading joint learner model.")
                    args.model = "joint"
                    model = JointLearner(
                        word_embedding_size,
                        lstm_hidden_size,
                        vocab,
                        MAX_CAPTION_LEN,
                        joint_embeddings_size,
                        fine_tune_resnet=False,
                    )

                else:
                    raise RuntimeError(f"Non supported model: {checkpoint_path}")

                model.load_state_dict(checkpoint["model_state_dict"])

                model = model.to(device)

                (
                    test_loss,
                    accuracies,
                    captioning_loss,
                    ranking_loss,
                    test_acc,
                    test_bleu_score,
                ) = validate_model(
                    model,
                    test_loader,
                    semantics_eval_loaders,
                    vocab,
                    args,
                    val_bleu_score=True,
                    max_batches=None
                )
                print(f"BLEU: {test_bleu_score}")
                bleu_scores.append(test_bleu_score)

    print(f"\nMean BLEU: {np.mean(bleu_scores):.4f} Stddev: {np.std(bleu_scores):.4f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints-dir", type=str,
    )
    parser.add_argument(
        "--eval-semantics",
        default=False,
        action="store_true",
        help="Eval semantics of model using 2AFC",
    )
    parser.add_argument(
        "--produced-utterances-stats",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--weights-bleu",
        default=(0.25, 0.25, 0.25, 0.25),
        type=float,
        nargs=4,
        help="Weights for BLEU score that is used as reward for RL",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed", default=1,
    )


    return parser.parse_args()


if __name__ == "__main__":
    print("Start test on device: ", device)
    args = get_args()
    main(args)
