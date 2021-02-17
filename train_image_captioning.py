
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
from torch.utils.data import DataLoader

import egg.core as core
from dataset import CaptionDataset
from models import ImageCaptioner
from preprocess import IMAGES_FILENAME, CAPTIONS_FILENAME, VOCAB_FILENAME, MAX_CAPTION_LEN, \
    DATA_PATH
from utils import print_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH_IMAGE_CAPTIONING = os.path.join(Path.home(), "data/egg/visual_ref/checkpoints/image_captioning.pt")


PRINT_SAMPLE_CAPTIONS = 5


def print_model_output(output, target_captions, image_ids, vocab, num_captions=1):
    captions_model = torch.argmax(output, dim=1)
    for i in range(num_captions):
        print(f"Image ID: {image_ids[i]}")
        print("Target: ", end="")
        print_caption(target_captions[i], vocab)
        print("Model output: ", end="")
        print_caption(captions_model[i], vocab)


def print_sample_model_output(model, dataloader, vocab, num_captions=1):
    images, captions, caption_lengths, image_ids = next(iter(dataloader))

    output, decode_lengths = model.forward_decode(images, decode_type="sample")

    print_model_output(output, captions, image_ids, vocab, num_captions)


def main(args):
    # create model checkpoint directory
    if not os.path.exists(os.path.dirname(CHECKPOINT_PATH_IMAGE_CAPTIONING)):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH_IMAGE_CAPTIONING))

    train_loader = DataLoader(
        CaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME["train"],
            CAPTIONS_FILENAME["train"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )
    val_images_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME["val"],
            CAPTIONS_FILENAME["val"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    word_embedding_size = 512
    visual_embedding_size = 512
    lstm_hidden_size = 512
    model_image_captioning = ImageCaptioner(word_embedding_size, visual_embedding_size, lstm_hidden_size, vocab,
                                            MAX_CAPTION_LEN, fine_tune_resnet=args.fine_tune_resnet)

    # uses command-line parameters we passed to core.init
    optimizer = core.build_optimizer(model_image_captioning.parameters())

    model_image_captioning = model_image_captioning.to(device)

    def save_model(model, optimizer, best_val_loss, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, CHECKPOINT_PATH_IMAGE_CAPTIONING)

    def validate_model(model, dataloader):
        print(f"EVAL")
        model.eval()
        with torch.no_grad():
            print_sample_model_output(model, dataloader, vocab, PRINT_SAMPLE_CAPTIONS)

            val_losses = []
            for batch_idx, (images, captions, caption_lengths, _) in enumerate(dataloader):
                output, decode_lengths = model.forward_decode(images, decode_type="sample")

                loss = model.calc_loss(output, captions, caption_lengths)

                val_losses.append(loss.mean().item())

            val_loss = np.mean(val_losses)

        model.train()
        return val_loss

    best_val_loss = math.inf
    for epoch in range(args.n_epochs):
        losses = []
        for batch_idx, (images, captions, caption_lengths, _) in enumerate(train_loader):
            output = model_image_captioning(images, captions, caption_lengths)

            loss = model_image_captioning.calc_loss(output, captions, caption_lengths)
            losses.append(loss.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_frequency == 0:
                val_loss = validate_model(model_image_captioning, val_images_loader)
                print(f"Batch {batch_idx}: train loss: {np.mean(losses)} | val loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model_image_captioning, optimizer, best_val_loss, epoch)

        val_loss = validate_model(model_image_captioning, val_images_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model_image_captioning, optimizer, best_val_loss, epoch)
        print(f'End of epoch: {epoch} | train loss: {np.mean(losses)} | best val loss: {best_val_loss}\n\n')

    core.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine-tune-resnet",
        default=False,
        action="store_true",
        help="Fine tune the ResNet module.",
    )
    parser.add_argument(
        "--log-frequency",
        default=100,
        type=int,
        help="Logging frequency (number of batches)",
    )

    # initialize the egg lib
    # get pre-defined common line arguments (batch/vocab size, etc).
    # See egg/core/util.py for a list
    args = core.init(parser)

    return args


if __name__ == "__main__":
    print("Start training on device: ", device)
    args = get_args()
    main(args)

