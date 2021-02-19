from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

import torch
import torch.distributions
import torch.utils.data

import egg.core as core
from dataset import SyntaxEvalDataset
from models.image_captioning.show_attend_and_tell import Show_Attend_And_Tell
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH,
)
from train_image_captioning import (
    CHECKPOINT_PATH_IMAGE_CAPTIONING_BEST,
)
from utils import decode_caption, show_image

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
    dropout = 0.2
    model = Show_Attend_And_Tell(word_embedding_size, lstm_hidden_size, vocab, MAX_CAPTION_LEN, dropout,
                                 fine_tune_resnet=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    # TODO fix for batching
    test_images_loader = torch.utils.data.DataLoader(
        SyntaxEvalDataset(DATA_PATH, IMAGES_FILENAME["test"], CAPTIONS_FILENAME["test"], args.eval_csv, vocab),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = model.to(device)
    model.eval()

    accuracies = []
    with torch.no_grad():
        for batch_idx, (img, target_caption, distractor_caption) in enumerate(
            test_images_loader
        ):
            images = torch.cat((img, img))
            captions = torch.cat((target_caption, distractor_caption))
            caption_lengths = torch.tensor([target_caption.shape[1], distractor_caption.shape[1]])

            perplexities = model.perplexity(images, captions, caption_lengths)

            print(f"Target    : {decode_caption(target_caption[0], vocab)}")
            print(f"Distractor: {decode_caption(distractor_caption[0], vocab)}")

            print(f"Perplexity target    : {perplexities[0]}")
            print(f"Perplexity distractor: {perplexities[1]}")

            if perplexities[0] < perplexities[1]:
                accuracies.append(1)
            else:
                accuracies.append(0)

    print(f"\n\n\nAccuracy: {np.mean(accuracies)}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH_IMAGE_CAPTIONING_BEST, type=str,
    )
    parser.add_argument(
        "--eval-csv", default="data/syntax_eval_agent_vs_patient.csv", type=str,
    )

    return core.init(parser)


if __name__ == "__main__":
    print("Start eval on device: ", device)
    args = get_args()
    main(args)
