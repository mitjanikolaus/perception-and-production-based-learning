from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

import torch
import torch.distributions
import torch.utils.data

from dataset import SemanticsEvalDataset
from models.image_captioning.show_and_tell import ShowAndTell
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker, cosine_sim
from models.image_sentence_ranking.ranking_model_grounded import (
    ImageSentenceRankerGrounded,
)
from models.interactive.models import RnnSenderMultitaskVisualRef, ImageEncoder
from models.joint.joint_learner import JointLearner
from models.joint.joint_learner_sat import JointLearnerSAT
from models.language_modeling.language_model import LanguageModel
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH,
)
from utils import decode_caption, SEMANTICS_EVAL_FILES, DEFAULT_WORD_EMBEDDINGS_SIZE, DEFAULT_LSTM_HIDDEN_SIZE, LEGEND

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_MAX_SAMPLES = 500


def get_semantics_eval_dataloader(eval_file, vocab):
    return torch.utils.data.DataLoader(
        SemanticsEvalDataset(
            DATA_PATH,
            IMAGES_FILENAME["test"],
            CAPTIONS_FILENAME["test"],
            eval_file,
            vocab,
        ),
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
            caption_lengths = torch.tensor(
                [target_caption.shape[1], distractor_caption.shape[1]], device=device
            )

            if verbose:
                print(f"Target    : {decode_caption(target_caption[0], vocab)}")
                print(f"Distractor: {decode_caption(distractor_caption[0], vocab)}")

            if (
                isinstance(model, ShowAttendAndTell)
                or isinstance(model, ShowAndTell)
                or isinstance(model, RnnSenderMultitaskVisualRef)
            ):
                perplexities = model.perplexity(images, captions, caption_lengths)

                if verbose:
                    print(f"Perplexity target    : {perplexities[0]}")
                    print(f"Perplexity distractor: {perplexities[1]}")

                if perplexities[0] < perplexities[1]:
                    accuracies.append(1)
                elif perplexities[0] > perplexities[1]:
                    accuracies.append(0)

            elif isinstance(model, ImageSentenceRanker) or isinstance(
                model, ImageSentenceRankerGrounded
            ):
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

            elif isinstance(model, JointLearner) or isinstance(model, JointLearnerSAT):
                _, _, _, images_embedded, captions_embedded = model(
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

            elif isinstance(model, LanguageModel):
                perplexities = model.perplexity(captions, caption_lengths)

                if verbose:
                    print(f"Perplexity target    : {perplexities[0]}")
                    print(f"Perplexity distractor: {perplexities[1]}")

                if perplexities[0] < perplexities[1]:
                    accuracies.append(1)
                elif perplexities[0] > perplexities[1]:
                    accuracies.append(0)

            else:
                raise RuntimeError(f"Unknown model: {model}")

            if len(accuracies) > EVAL_MAX_SAMPLES:
                break

    return np.mean(accuracies)


def main(args):
    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    # Initialize lists for storing results of multiple checkpoints
    semantic_accuracies = {key: [] for key in SEMANTICS_EVAL_FILES}

    for checkpoint_dir in args.checkpoints:
        print("Loading model checkpoint from {}".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location=device)

        word_embedding_size = DEFAULT_WORD_EMBEDDINGS_SIZE
        joint_embeddings_size = DEFAULT_LSTM_HIDDEN_SIZE
        lstm_hidden_size = DEFAULT_LSTM_HIDDEN_SIZE
        visual_embedding_size = DEFAULT_LSTM_HIDDEN_SIZE

        if "show_attend_and_tell" in checkpoint_dir:
            print("Loading sat image captioning model.")
            word_embedding_size = 512
            lstm_hidden_size = 512
            model = ShowAttendAndTell(
                word_embedding_size,
                lstm_hidden_size,
                vocab,
                MAX_CAPTION_LEN,
                fine_tune_resnet=False,
            )

        elif "show_and_tell" in checkpoint_dir:
            print("Loading st image captioning model.")
            model = ShowAndTell(
                word_embedding_size,
                visual_embedding_size,
                lstm_hidden_size,
                vocab,
                MAX_CAPTION_LEN,
                fine_tune_resnet=False,
            )

        elif "sender" in checkpoint_dir:
            print("Loading sender image captioning model.")
            word_embedding_size = DEFAULT_WORD_EMBEDDINGS_SIZE
            sender_hidden = DEFAULT_LSTM_HIDDEN_SIZE
            encoder = ImageEncoder(sender_hidden, fine_tune_resnet=False)
            model = RnnSenderMultitaskVisualRef(
                encoder,
                vocab=vocab,
                embed_dim=word_embedding_size,
                hidden_size=sender_hidden,
                cell="lstm",
                max_len=MAX_CAPTION_LEN,
            )

        elif "joint" in checkpoint_dir:
            print("Loading joint learner model.")
            model = JointLearner(
                word_embedding_size,
                lstm_hidden_size,
                vocab,
                MAX_CAPTION_LEN,
                joint_embeddings_size,
                fine_tune_resnet=False,
            )

        elif "ranking" in checkpoint_dir:
            print("Loading image sentence ranking model.")
            model = ImageSentenceRanker(
                word_embedding_size,
                joint_embeddings_size,
                lstm_hidden_size,
                len(vocab),
                fine_tune_resnet=False,
            )

        elif "language_model" in checkpoint_dir:
            print("Loading language model.")
            model = LanguageModel(word_embedding_size, lstm_hidden_size, vocab)

        else:
            raise RuntimeError(f"Unknown model: {args.checkpoint}")

        model.load_state_dict(checkpoint["model_state_dict"])

        model = model.to(device)

        semantics_eval_loaders = {
            file: get_semantics_eval_dataloader(file, vocab)
            for file in SEMANTICS_EVAL_FILES
        }

        for name, semantic_images_loader in semantics_eval_loaders.items():
            acc = eval_semantics_score(
                model, semantic_images_loader, vocab, verbose=args.verbose
            )
            print(f"Accuracy for {LEGEND[name]}: {acc:.3f}")
            semantic_accuracies[name].append(acc)

    print("\nAverage stats:")
    for name, accs in semantic_accuracies.items():
        print(f"Accuracy for {LEGEND[name]}: {np.mean(accs):.3f} (stddev: {np.std(accs):.3f})")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+", type=str,
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Start eval on device: ", device)
    args = get_args()
    main(args)
