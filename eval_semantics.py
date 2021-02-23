from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

import torch
import torch.distributions
import torch.utils.data

import egg.core as core
from dataset import SemanticsEvalDataset
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker, cosine_sim
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
from utils import decode_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    print("Loading model checkpoint from {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "image_captioning" in args.checkpoint:
        print("Loading image captioning model.")
        word_embedding_size = 512
        visual_embedding_size = 512
        lstm_hidden_size = 512
        dropout = 0.2
        model = ShowAttendAndTell(word_embedding_size, lstm_hidden_size, vocab, MAX_CAPTION_LEN, dropout,
                                  fine_tune_resnet=False)

    elif "ranking" in args.checkpoint:
        print('Loading image sentence ranking model.')
        word_embedding_size = 100
        joint_embeddings_size = 512
        lstm_hidden_size = 512
        model = ImageSentenceRanker(
            word_embedding_size,
            joint_embeddings_size,
            lstm_hidden_size,
            len(vocab),
            fine_tune_resnet=False,
        )
    else:
        raise RuntimeError(f"Unknown model: {args.checkpoint}")

    model.load_state_dict(checkpoint["model_state_dict"])

    # TODO fix for batching
    test_images_loader = torch.utils.data.DataLoader(
        SemanticsEvalDataset(DATA_PATH, IMAGES_FILENAME["test"], CAPTIONS_FILENAME["test"], args.eval_csv, vocab),
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

            print(f"Target    : {decode_caption(target_caption[0], vocab)}")
            print(f"Distractor: {decode_caption(distractor_caption[0], vocab)}")

            if isinstance(model, ShowAttendAndTell):
                perplexities = model.perplexity(images, captions, caption_lengths)

                print(f"Perplexity target    : {perplexities[0]}")
                print(f"Perplexity distractor: {perplexities[1]}")

                if perplexities[0] < perplexities[1]:
                    accuracies.append(1)
                else:
                    accuracies.append(0)
            else:
                # Assuming ranking model
                images_embedded, captions_embedded = model(
                    images, captions, caption_lengths
                )

                similarities = cosine_sim(images_embedded, captions_embedded)[0]

                print(f"Similarity target    : {similarities[0]}")
                print(f"Similarity distractor: {similarities[1]}")

                if similarities[0] > similarities[1]:
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
        "--eval-csv", default="data/semantics_eval_actors.csv", type=str,
    )

    return core.init(parser)


if __name__ == "__main__":
    print("Start eval on device: ", device)
    args = get_args()
    main(args)
