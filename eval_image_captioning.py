from __future__ import print_function

import argparse
import pickle
import os


import torch
import torch.distributions
import torch.utils.data

import egg.core as core
from dataset import CaptionDataset, SyntaxEvalDataset
from models import ImageCaptioner
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH,
)
from train_image_captioning import (
    CHECKPOINT_PATH_IMAGE_CAPTIONING,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    print("Loading model checkpoint from {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    word_embedding_size = 512
    visual_embedding_size = 512
    lstm_hidden_size = 512
    model = ImageCaptioner(
        word_embedding_size,
        visual_embedding_size,
        lstm_hidden_size,
        vocab,
        MAX_CAPTION_LEN,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_images_loader = torch.utils.data.DataLoader(
        SyntaxEvalDataset(DATA_PATH, IMAGES_FILENAME["test"], CAPTIONS_FILENAME["test"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for batch_idx, (img_target, img_distractor, caption, caption_length) in enumerate(
            test_images_loader
        ):
            images = torch.cat((img_target, img_distractor))
            captions = torch.cat((caption, caption))
            caption_lengths = torch.tensor([caption_length, caption_length])

            perplexities = model.perplexity(images, captions, caption_lengths)
            print(f"Perplexity target: {perplexities[0]}")
            print(f"Perplexity distra: {perplexities[1]}")

    core.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH_IMAGE_CAPTIONING, type=str,
    )

    return core.init(parser)


if __name__ == "__main__":
    print("Start eval on device: ", device)
    args = get_args()
    main(args)
