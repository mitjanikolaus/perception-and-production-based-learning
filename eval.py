#  python -m train --vocab_size=10 --n_epochs=15 --random_seed=7 --lr=1e-3 --batch_size=32 --optimizer=adam
import argparse
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader

from torch.nn import functional as F

import matplotlib.pyplot as plt

import egg.core as core
from egg.core import ConsoleLogger, Callback, Interaction
from dataset import VisualRefGameDataset
from game import OracleSenderReceiverRnnSupervised
from preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
)
from train import loss_multitask
from utils import decode_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    train_dataset = VisualRefGameDataset(
        DATA_PATH, IMAGES_FILENAME["train"], args.batch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_dataset = VisualRefGameDataset(
        DATA_PATH, IMAGES_FILENAME["val"], args.batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    # TODO
    # TODO: embedding size for speaker is 1024 in paper
    args.sender_hidden = 1024  # TODO
    args.sender_embedding = 512  # ???
    args.receiver_embedding = 100  # ???
    args.receiver_hidden = 512  # ???
    args.sender_entropy_coeff = 0.0  # entropy regularization
    args.receiver_entropy_coeff = 0.0  # entropy regularization
    args.sender_cell = "lstm"
    args.receiver_cell = "lstm"
    args.vocab_size = len(vocab)
    args.max_len = 25
    args.random_seed = 1

    word_embedding_size = 100
    joint_embeddings_size = 512
    lstm_hidden_size = 512
    checkpoint_ranking_model = torch.load(
        CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING, map_location=device
    )
    ranking_model = ImageSentenceRanker(
        word_embedding_size,
        joint_embeddings_size,
        lstm_hidden_size,
        len(vocab),
        fine_tune_resnet=False,
    )
    ranking_model.load_state_dict(checkpoint_ranking_model["model_state_dict"])

    sender = VisualRefSpeakerDiscriminativeOracle(
        DATA_PATH, CAPTIONS_FILENAME, args.max_len, vocab
    )
    receiver = VisualRefListenerOracle(ranking_model)

    # use LoggingStrategy that stores image IDs
    logging_strategy = VisualRefLoggingStrategy()

    game = OracleSenderReceiverRnnSupervised(
        sender,
        receiver,
        loss_multitask,
        receiver_entropy_coeff=args.receiver_entropy_coeff,
        train_logging_strategy=logging_strategy,
        test_logging_strategy=logging_strategy,
    )

    callbacks = [ConsoleLogger(print_train_loss=True, as_json=False)]

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
    )

    print("Starting eval with args: ")
    print(args)
    print("Number of samples: ", len(val_dataset))
    game.eval()
    val_loss, interactions = trainer.eval()

    print(f"Val loss: {val_loss:.3f}")
    print(f"Val acc: {interactions.aux['acc'].mean():.3f}")

    core.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Print debug interactions output.",
    )
    parser.add_argument(
        "--log-frequency",
        default=100,
        type=int,
        help="Logging frequency (number of batches)",
    )
    args = core.init(parser)

    return args


if __name__ == "__main__":
    print("Start eval on device: ", device)
    args = get_args()
    main(args)
