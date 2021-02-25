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
from utils import decode_caption, CHECKPOINT_PATH_IMAGE_CAPTIONING_BEST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_MAX_SAMPLES = 100

def get_semantics_eval_dataloader(eval_file, vocab):
    return torch.utils.data.DataLoader(
        SemanticsEvalDataset(DATA_PATH, IMAGES_FILENAME["test"], CAPTIONS_FILENAME["test"],
                             eval_file, vocab),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )


def eval_semantics_score(model, dataloader, vocab, verbose=False):
    model.eval()

    accuracies = []
    with torch.no_grad():
        for batch_idx, (img, target_caption, distractor_caption) in enumerate(
                dataloader
        ):
            images = torch.cat((img, img))
            captions = torch.cat((target_caption, distractor_caption))
            caption_lengths = torch.tensor([target_caption.shape[1], distractor_caption.shape[1]], device=device)

            if verbose:
                print(f"Target    : {decode_caption(target_caption[0], vocab)}")
                print(f"Distractor: {decode_caption(distractor_caption[0], vocab)}")

            if isinstance(model, ShowAttendAndTell):
                perplexities = model.perplexity(images, captions, caption_lengths)

                if verbose:
                    print(f"Perplexity target    : {perplexities[0]}")
                    print(f"Perplexity distractor: {perplexities[1]}")

                if perplexities[0] < perplexities[1]:
                    accuracies.append(1)
                elif perplexities[0] > perplexities[1]:
                    accuracies.append(0)
            else:
                # Assuming ranking model
                images_embedded, captions_embedded = model(
                    images, captions, caption_lengths
                )

                similarities = cosine_sim(images_embedded, captions_embedded)[0]

                if verbose:
                    print(f"Similarity target    : {similarities[0]}")
                    print(f"Similarity distractor: {similarities[1]}")

                if similarities[0] > similarities[1]:
                    accuracies.append(1)
                elif similarities[0] < similarities[1]:
                    accuracies.append(0)

            if len(accuracies) > EVAL_MAX_SAMPLES:
                break

    return np.mean(accuracies)


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

    test_images_loader = get_semantics_eval_dataloader(args.eval_csv, vocab)

    model = model.to(device)

    eval_semantics_score(model, test_images_loader, vocab, verbose=True)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH_IMAGE_CAPTIONING_BEST, type=str,
    )
    parser.add_argument(
        "--eval-csv", default="data/semantics_eval_persons.csv", type=str,
    )

    return core.init(parser)


if __name__ == "__main__":
    print("Start eval on device: ", device)
    args = get_args()
    main(args)
