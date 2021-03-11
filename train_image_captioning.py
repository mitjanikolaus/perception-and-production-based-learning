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
from models.image_captioning.show_and_tell import ShowAndTell
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.image_sentence_ranking.ranking_model import accuracy_discrimination
from models.joint.joint_learner import JointLearner
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH,
)
from utils import (
    print_caption,
    CHECKPOINT_DIR_IMAGE_CAPTIONING,
    SEMANTICS_EVAL_FILES, DEFAULT_LOG_FREQUENCY,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRINT_SAMPLE_CAPTIONS = 1

NUM_BATCHES_VALIDATION = 10

WEIGH_RANKING_LOSS = 100


def print_model_output(output, target_captions, image_ids, vocab, num_captions=1):
    captions_model = torch.argmax(output, dim=1)
    print_captions(captions_model, target_captions, image_ids, vocab, num_captions)


def print_captions(captions, target_captions, image_ids, vocab, num_captions=1):
    for i in range(num_captions):
        print(f"Image ID: {image_ids[i]}")
        print("Target: ", end="")
        print_caption(target_captions[i], vocab)
        print("Model output: ", end="")
        print_caption(captions[i], vocab)


def print_sample_model_output(model, dataloader, vocab, num_captions=1):
    images, target_captions, caption_lengths, image_ids = next(iter(dataloader))

    captions, _, _ = model.decode_nucleus_sampling(images, 1, top_p=0.9)

    print_captions(captions, target_captions, image_ids, vocab, num_captions)


def validate_model(
    model, dataloader, print_images_loader, semantic_images_loaders, vocab
):
    semantic_accuracies = {}

    model.eval()
    with torch.no_grad():
        print_sample_model_output(
            model, print_images_loader, vocab, PRINT_SAMPLE_CAPTIONS
        )
        for name, semantic_images_loader in semantic_images_loaders.items():
            acc = eval_semantics_score(model, semantic_images_loader, vocab)
            print(f"Accuracy for {name}: {acc:.3f}")
            semantic_accuracies[name] = acc

        val_losses = []
        captioning_losses = []
        ranking_losses = []
        val_accuracies = []
        for batch_idx, (images, captions, caption_lengths, _) in enumerate(dataloader):
            if args.model == "joint":
                scores, decode_lengths, alphas, images_embedded, captions_embedded = model(
                    images, captions, caption_lengths
                )
                loss_captioning, loss_ranking = model.loss(scores, captions, decode_lengths, alphas, images_embedded, captions_embedded)

                # TODO weigh losses
                loss_ranking = WEIGH_RANKING_LOSS * loss_ranking
                loss = loss_captioning + loss_ranking

                acc = accuracy_discrimination(images_embedded, captions_embedded)
                val_accuracies.append(acc)

                captioning_losses.append(loss_captioning.item())
                ranking_losses.append(loss_ranking.item())
            else:
                scores, decode_lengths, alphas = model(images, captions, caption_lengths)
                loss = model.loss(scores, captions, decode_lengths, alphas)

            val_losses.append(loss.mean().item())

            if batch_idx > NUM_BATCHES_VALIDATION:
                break

    model.train()
    return np.mean(val_losses), semantic_accuracies, np.mean(captioning_losses), np.mean(ranking_losses), np.mean(val_accuracies)


def main(args):
    # create model checkpoint directory
    if not os.path.exists(os.path.dirname(CHECKPOINT_DIR_IMAGE_CAPTIONING)):
        os.makedirs(os.path.dirname(CHECKPOINT_DIR_IMAGE_CAPTIONING))

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    train_loader = DataLoader(
        CaptionDataset(
            DATA_PATH, IMAGES_FILENAME["train"], CAPTIONS_FILENAME["train"], vocab,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )
    val_images_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"], vocab),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )

    # TODO
    print_captions_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"], vocab),
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

    word_embedding_size = 128
    visual_embedding_size = 512
    joint_embeddings_size = visual_embedding_size
    lstm_hidden_size = 512
    dropout = 0.2

    if args.model == "show_attend_and_tell":
        model = ShowAttendAndTell(
            word_embedding_size,
            lstm_hidden_size,
            vocab,
            MAX_CAPTION_LEN,
            dropout,
            fine_tune_resnet=args.fine_tune_resnet,
        )
    elif args.model == "show_and_tell":
        word_embedding_size = 512
        model = ShowAndTell(
            word_embedding_size,
            visual_embedding_size,
            lstm_hidden_size,
            vocab,
            MAX_CAPTION_LEN,
            dropout,
            fine_tune_resnet=args.fine_tune_resnet,
        )
    elif args.model == "joint":
        word_embedding_size = 512
        model = JointLearner(
            word_embedding_size,
            lstm_hidden_size,
            vocab,
            MAX_CAPTION_LEN,
            joint_embeddings_size,
            dropout,
            fine_tune_resnet=args.fine_tune_resnet,
        )
    else:
        raise RuntimeError(f"Unknown model: ", args.model)

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
    accuracies_over_time = []
    for epoch in range(args.n_epochs):
        losses = []
        for batch_idx, (images, captions, caption_lengths, _) in enumerate(
            train_loader
        ):
            if batch_idx % args.log_frequency == 0:
                val_loss, semantic_accuracies, captioning_loss, ranking_loss, val_acc = validate_model(
                    model,
                    val_images_loader,
                    print_captions_loader,
                    semantics_eval_loaders,
                    vocab,
                )
                semantic_accuracies["val_loss"] = val_loss
                accuracies_over_time.append(semantic_accuracies)
                pickle.dump(
                    accuracies_over_time,
                    open(CHECKPOINT_DIR_IMAGE_CAPTIONING+args.model+"_accuracies.p", "wb"),
                )
                print(
                    f"Batch {batch_idx}: train loss: {np.mean(losses):.3f} | val loss: {val_loss:.3f} | captioning loss:"
                    f" {captioning_loss:.3f} | ranking loss: {ranking_loss:.3f} | val acc: {val_acc:.3f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(
                        model,
                        optimizer,
                        best_val_loss,
                        epoch,
                        CHECKPOINT_DIR_IMAGE_CAPTIONING+args.model+".pt",
                    )

            model.train()

            # Forward pass
            if args.model == "joint":
                scores, decode_lengths, alphas, images_embedded, captions_embedded = model(
                    images, captions, caption_lengths
                )
                loss_captioning, loss_ranking = model.loss(scores, captions, decode_lengths, alphas, images_embedded, captions_embedded)
                # TODO weigh losses
                loss = loss_captioning + loss_ranking
            else:
                scores, decode_lengths, alphas = model(images, captions, caption_lengths)
                loss = model.loss(scores, captions, decode_lengths, alphas)

            losses.append(loss.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, semantic_accuracies, _, _, _ = validate_model(
            model,
            val_images_loader,
            print_captions_loader,
            semantics_eval_loaders,
            vocab,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model,
                optimizer,
                best_val_loss,
                epoch,
                CHECKPOINT_DIR_IMAGE_CAPTIONING+args.model+".pt",
            )

        print(
            f"End of epoch: {epoch} | train loss: {np.mean(losses)} | best val loss: {best_val_loss}\n\n"
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="show_attend_and_tell",
        choices=["show_and_tell", "show_attend_and_tell", "joint"],
    )
    parser.add_argument(
        "--fine-tune-resnet",
        default=False,
        action="store_true",
        help="Fine tune the ResNet module.",
    )
    parser.add_argument(
        "--log-frequency",
        default=DEFAULT_LOG_FREQUENCY,
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
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Start training on device: ", device)
    args = get_args()
    main(args)
