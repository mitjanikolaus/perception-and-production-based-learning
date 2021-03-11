from __future__ import print_function

import argparse
import math
import pickle
import os

import numpy as np

import torch
import torch.distributions
import torch.utils.data
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CaptionDataset
from eval_semantics import get_semantics_eval_dataloader, eval_semantics_score
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker, accuracy_discrimination
from models.image_sentence_ranking.ranking_model_grounded import ImageSentenceRankerGrounded
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    DATA_PATH,
)
from utils import SEMANTICS_EVAL_FILES, CHECKPOINT_DIR_RANKING, DEFAULT_LOG_FREQUENCY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_model(model, dataloader, semantic_images_loaders, vocab, args):
    print(f"EVAL")
    semantic_accuracies = {}
    model.eval()
    with torch.no_grad():
        for name, semantic_images_loader in semantic_images_loaders.items():
            acc = eval_semantics_score(model, semantic_images_loader, vocab)
            print(f"Accuracy for {name}: {acc:.3f}")
            semantic_accuracies[name] = acc

        val_losses = []
        val_accuracies = []
        for batch_idx, (images, captions, caption_lengths, _) in enumerate(dataloader):
            if args.alternative_objective:
                for image in images:
                    image_repeated = image.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
                    images_embedded, captions_embedded = model(
                        image_repeated, captions, caption_lengths
                    )

                    acc = accuracy_discrimination(images_embedded, captions_embedded)
                    val_accuracies.append(acc)

                    loss = model.loss(images_embedded, captions_embedded)
                    val_losses.append(loss.item())
            else:
                images_embedded, captions_embedded = model(
                    images, captions, caption_lengths
                )

                acc = accuracy_discrimination(images_embedded, captions_embedded)
                val_accuracies.append(acc)

                loss = model.loss(images_embedded, captions_embedded)
                val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accuracies)

    return val_loss, val_acc, semantic_accuracies


def main(args):
    # create model checkpoint directory
    if not os.path.exists(os.path.dirname(args.checkpoint_dir)):
        os.makedirs(os.path.dirname(args.checkpoint_dir))

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

    semantics_eval_loaders = {
        file: get_semantics_eval_dataloader(file, vocab) for file in SEMANTICS_EVAL_FILES
    }

    word_embedding_size = 100
    joint_embeddings_size = 512
    lstm_hidden_size = 512

    if args.alternative_objective:
        model = ImageSentenceRankerGrounded(
            word_embedding_size,
            joint_embeddings_size,
            lstm_hidden_size,
            len(vocab),
            fine_tune_resnet=args.fine_tune_resnet,
        )
    else:
        model = ImageSentenceRanker(
            word_embedding_size,
            joint_embeddings_size,
            lstm_hidden_size,
            len(vocab),
            fine_tune_resnet=args.fine_tune_resnet,
        )

    optimizer = Adam(model.parameters(), lr=args.lr)

    model = model.to(device)

    def save_model(model, optimizer, best_val_loss, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            },
            args.checkpoint_dir+"/ranking.pt",
        )

    best_val_loss = math.inf
    accuracies_over_time = []
    for epoch in range(args.n_epochs):
        losses = []
        for batch_idx, (images, captions, caption_lengths, _) in enumerate(train_loader):
            if batch_idx % args.log_frequency == 0:
                val_loss, val_acc, semantic_accuracies = validate_model(model, val_images_loader, semantics_eval_loaders, vocab, args)
                print(f"Batch {batch_idx}: train loss: {np.mean(losses)} val loss: {val_loss} val acc: {val_acc}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model, optimizer, best_val_loss, epoch)
                semantic_accuracies["val_loss"] = val_loss
                accuracies_over_time.append(semantic_accuracies)
                pickle.dump(accuracies_over_time, open(args.checkpoint_dir+"/ranking_accuracies.p", "wb"))

            model.train()

            if args.alternative_objective:
                for image in images:
                    image_repeated = image.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
                    images_embedded, captions_embedded = model(
                        image_repeated, captions, caption_lengths
                    )

                    loss = model.loss(images_embedded, captions_embedded)
                    losses.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:
                images_embedded, captions_embedded = model(
                    images, captions, caption_lengths
                )

                loss = model.loss(images_embedded, captions_embedded)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        val_loss, _, _ = validate_model(model, val_images_loader, semantics_eval_loaders, vocab, args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, best_val_loss, epoch)

        print(f"Train Epoch: {epoch}, train loss: {np.mean(losses)} best val loss: {best_val_loss}\n\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=CHECKPOINT_DIR_RANKING,
        type=str,
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
    parser.add_argument(
        "--alternative-objective",
        default=False,
        action="store_true",
        help="Use the alternative objective (with grounded language encoder).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Start training on device: ", device)
    main(get_args())
