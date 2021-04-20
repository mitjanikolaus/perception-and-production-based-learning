from __future__ import print_function

import argparse
import math
import pickle
import sys
from pathlib import Path
import os

import numpy as np

import torch
import torch.distributions
import torch.utils.data
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CaptionDataset, SemanticsEvalDataset
from eval_semantics import eval_semantics_score, get_semantics_eval_dataloader
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.language_modeling.language_model import LanguageModel
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH,
)
from train_image_captioning import print_sample_model_output
from utils import (
    print_caption,
    SEMANTICS_EVAL_FILES,
    CHECKPOINT_PATH_LANGUAGE_MODEL_BEST,
    SEMANTIC_ACCURACIES_PATH_LANGUAGE_MODEL,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRINT_SAMPLE_CAPTIONS = 1

NUM_BATCHES_VALIDATION = 10


def validate_model(
    model, dataloader, print_images_loader, semantic_images_loaders, vocab
):
    semantic_accuracies = {}

    model.eval()
    with torch.no_grad():
        for name, semantic_images_loader in semantic_images_loaders.items():
            acc = eval_semantics_score(model, semantic_images_loader, vocab)
            print(f"Accuracy for {name}: {acc}")
            semantic_accuracies[name] = acc

        val_losses = []
        for batch_idx, (_, captions, caption_lengths, _) in enumerate(dataloader):
            scores = model(captions, caption_lengths)

            loss = model.loss(scores, captions)

            val_losses.append(loss.mean().item())

            if batch_idx > NUM_BATCHES_VALIDATION:
                break

    model.train()
    return np.mean(val_losses), semantic_accuracies


def main(args):
    # create model checkpoint directory
    if not os.path.exists(os.path.dirname(CHECKPOINT_PATH_LANGUAGE_MODEL_BEST)):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH_LANGUAGE_MODEL_BEST))

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    train_loader = DataLoader(
        CaptionDataset(
            DATA_PATH, IMAGES_FILENAME["train"], CAPTIONS_FILENAME["train"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"],),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )

    # TODO
    print_captions_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"],),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )

    semantics_eval_loaders = {
        file: get_semantics_eval_dataloader(file, vocab)
        for file in SEMANTICS_EVAL_FILES
    }

    word_embedding_size = 512
    lstm_hidden_size = 512
    dropout = 0.2
    model = LanguageModel(word_embedding_size, lstm_hidden_size, vocab, dropout,)

    optimizer = Adam(model.parameters(), lr=args.lr)

    model = model.to(device)

    def save_model(model, optimizer, best_val_loss, epoch, path):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            },
            path,
        )

    best_val_loss = math.inf
    semantic_accuracies_over_time = []
    for epoch in range(args.n_epochs):
        losses = []
        for batch_idx, (_, captions, caption_lengths, _) in enumerate(train_loader):
            if batch_idx % args.log_frequency == 0:
                val_loss, semantic_accuracies = validate_model(
                    model,
                    val_loader,
                    print_captions_loader,
                    semantics_eval_loaders,
                    vocab,
                )
                semantic_accuracies_over_time.append(semantic_accuracies)
                pickle.dump(
                    semantic_accuracies_over_time,
                    open(SEMANTIC_ACCURACIES_PATH_LANGUAGE_MODEL, "wb"),
                )
                print(
                    f"Batch {batch_idx}: train loss: {np.mean(losses)} | val loss: {val_loss}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(
                        model,
                        optimizer,
                        best_val_loss,
                        epoch,
                        CHECKPOINT_PATH_LANGUAGE_MODEL_BEST,
                    )

            model.train()

            # Forward pass
            scores = model(captions, caption_lengths)

            loss = model.loss(scores, captions)
            losses.append(loss.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, semantic_accuracies = validate_model(
            model, val_loader, print_captions_loader, semantics_eval_loaders, vocab,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model,
                optimizer,
                best_val_loss,
                epoch,
                CHECKPOINT_PATH_LANGUAGE_MODEL_BEST,
            )

        print(
            f"End of epoch: {epoch} | train loss: {np.mean(losses)} | best val loss: {best_val_loss}\n\n"
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-frequency",
        default=100,
        type=int,
        help="Logging frequency (number of batches)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=15,
        help="Number of epochs to train (default: 15)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for training (default: 32)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")

    return parser.parse_args()


if __name__ == "__main__":
    print("Start training on device: ", device)
    args = get_args()
    main(args)
