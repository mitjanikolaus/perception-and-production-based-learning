from __future__ import print_function

import math
import pickle
import sys
from pathlib import Path
import os

import numpy as np

import torch
import torch.distributions
import torch.utils.data
from torch.utils.data import DataLoader

import egg.core as core
from dataset import CaptionDataset
from models import ImageSentenceRanker
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    DATA_PATH,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING = os.path.join(
    Path.home(), "data/egg/visual_ref/checkpoints/image_sentence_ranking.pt"
)


VAL_INTERVAL = 100


def main(params):
    # initialize the egg lib
    opts = core.init(params=params)

    # create model checkpoint directory
    if not os.path.exists(os.path.dirname(CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING)):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING))

    train_loader = DataLoader(
        CaptionDataset(
            DATA_PATH, IMAGES_FILENAME["train"], CAPTIONS_FILENAME["train"],
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )
    val_images_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"],),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    word_embedding_size = 100
    joint_embeddings_size = 512
    lstm_hidden_size = 512
    model_ranking = ImageSentenceRanker(
        word_embedding_size,
        joint_embeddings_size,
        lstm_hidden_size,
        len(vocab),
        fine_tune_resnet=False,
    )

    # uses command-line parameters we passed to core.init
    optimizer = core.build_optimizer(model_ranking.parameters())

    model_ranking = model_ranking.to(device)

    def save_model(model, optimizer, best_val_loss, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            },
            CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING,
        )

    def validate_model(model, dataloader):
        print(f"EVAL")
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_accuracies = []
            for batch_idx, (images, captions, caption_lengths, _) in enumerate(dataloader):
                images_embedded, captions_embedded = model_ranking(
                    images, captions, caption_lengths
                )

                acc = model_ranking.accuracy_discrimination(images_embedded, captions_embedded)
                val_accuracies.append(acc)

                loss = model_ranking.loss(images_embedded, captions_embedded)
                val_losses.append(loss.mean().item())

            val_loss = np.mean(val_losses)
            print(f"val loss: {val_loss}")

            val_acc = np.mean(val_accuracies)
            print(f"val acc: {val_acc}")

        model.train()
        return val_loss, val_acc

    best_val_loss = math.inf
    for epoch in range(opts.n_epochs):
        losses = []
        for batch_idx, (images, captions, caption_lengths, _) in enumerate(train_loader):
            images_embedded, captions_embedded = model_ranking(
                images, captions, caption_lengths
            )

            loss = model_ranking.loss(images_embedded, captions_embedded)
            losses.append(loss.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % VAL_INTERVAL == 0:
                print(f"Batch {batch_idx}: train loss: {np.mean(losses)}")
                val_loss, _ = validate_model(model_ranking, val_images_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model_ranking, optimizer, best_val_loss, epoch)

        print(f"Train Epoch: {epoch}, train loss: {np.mean(losses)}")
        val_loss, _ = validate_model(model_ranking, val_images_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model_ranking, optimizer, best_val_loss, epoch)

    core.close()


if __name__ == "__main__":
    print("Start training on device: ", device)
    main(sys.argv[1:])
