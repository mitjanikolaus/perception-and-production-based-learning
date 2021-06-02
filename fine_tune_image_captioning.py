import argparse
import math
import pathlib
import pickle
import os
import random
from collections import defaultdict

import numpy as np

import pandas as pd

import torch
import torch.distributions
import torch.utils.data
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CaptionDataset, CaptionRLDataset
from eval_semantics import eval_semantics_score, get_semantics_eval_dataloader
from models.image_captioning.show_and_tell import ShowAndTell
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.image_sentence_ranking.ranking_model import accuracy_discrimination
from models.interactive.models import (
    ImageEncoder,
    RnnSenderMultitaskVisualRef,
    loss_cross_entropy,
)
from models.joint.joint_learner import JointLearner
from models.joint.joint_learner_sat import JointLearnerSAT
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    MAX_CAPTION_LEN,
    DATA_PATH,
)
from train_image_captioning import validate_model, save_model, forward_pass
from utils import (
    print_caption,
    CHECKPOINT_DIR_IMAGE_CAPTIONING,
    SEMANTICS_EVAL_FILES,
    DEFAULT_LOG_FREQUENCY,
    DEFAULT_WORD_EMBEDDINGS_SIZE,
    DEFAULT_LSTM_HIDDEN_SIZE, set_seeds,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_pass_rl(model, images, captions, vocab, args):
    sequences, logits, entropies, sequence_lengths = model.decode(
        images, sampling=True
    )
    # # Do another forward pass with greedy decoding for baseline:
    # with torch.no_grad():
    #     model.eval()
    #     sequences_greedy, _, _, decode_lengths_greedy = model.decode(
    #         images, sampling=False
    #     )
    #     model.train()

    reward = model.reward_rl(
        sequences,
        captions,
        vocab
    )
    # reward_baseline = model.reward_rl(
    #     sequences_greedy,
    #     captions,
    #     vocab
    # )

    # TODO: check whether this step is superfluous
    # # the log prob/ entropy of the choices made by S before and including the eos symbol
    effective_entropy = torch.zeros(entropies.shape[0], device=device)
    effective_log_prob = torch.zeros(logits.shape[0], device=device)

    for i in range(max(sequence_lengths)):
        not_eosed = (i < sequence_lengths).float()
        effective_entropy += entropies[:, i] * not_eosed
        effective_log_prob += logits[:, i] * not_eosed
    effective_entropy = effective_entropy / sequence_lengths.float()
    effective_log_prob = effective_log_prob / sequence_lengths.float()

    weighted_entropy = effective_entropy.mean() * args.entropy_coeff

    length_loss = sequence_lengths.float() * args.length_cost

    policy_length_loss = (
            length_loss * effective_log_prob
    ).mean()
    policy_loss = - (
            (reward) * effective_log_prob
    ).mean()
    # policy_loss = - (
    #         (reward - reward_baseline) * effective_log_prob
    # ).mean()

    rl_loss = policy_length_loss + policy_loss - weighted_entropy

    return rl_loss, reward.mean()


def main(args):
    if args.seed:
        set_seeds(args.seed)

    # create model checkpoint directory
    if not os.path.exists(args.out_checkpoints_dir):
        os.makedirs(args.out_checkpoints_dir)

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    train_dataset = CaptionRLDataset(
        DATA_PATH, IMAGES_FILENAME["train"], CAPTIONS_FILENAME["train"], vocab, args.training_set_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionRLDataset.pad_collate,
    )
    val_images_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"], vocab
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionDataset.pad_collate,
    )

    semantics_eval_loaders = {
        file: get_semantics_eval_dataloader(file, vocab)
        for file in SEMANTICS_EVAL_FILES
    }

    word_embedding_size = DEFAULT_WORD_EMBEDDINGS_SIZE
    visual_embedding_size = DEFAULT_LSTM_HIDDEN_SIZE
    joint_embeddings_size = visual_embedding_size
    lstm_hidden_size = DEFAULT_LSTM_HIDDEN_SIZE
    dropout = 0.2

    if args.model == "interactive":
        raise NotImplementedError()

    elif args.model == "show_attend_and_tell":
        model = ShowAttendAndTell(
            word_embedding_size,
            lstm_hidden_size,
            vocab,
            MAX_CAPTION_LEN,
            dropout,
            fine_tune_resnet=args.fine_tune_resnet,
        )
    elif args.model == "show_and_tell":
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

    if args.checkpoint:
        print(f"Loading model checkpoint from: {args.checkpoint}")
        model_checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(model_checkpoint["model_state_dict"])

    model = model.to(device)

    best_val_loss = math.inf
    accuracies_over_time = []
    for epoch in range(args.n_epochs):
        losses = []
        losses_rl = []
        losses_supervised = []
        bleu_scores = []

        for batch_idx, (images, captions, caption_lengths, _) in enumerate(
            train_loader
        ):
            if batch_idx % args.log_frequency == 0:
                (
                    val_loss,
                    accuracies,
                    captioning_loss,
                    ranking_loss,
                    val_acc,
                ) = validate_model(
                    model, val_images_loader, semantics_eval_loaders, vocab, args
                )
                accuracies["val_loss"] = val_loss
                accuracies["batch_id"] = batch_idx
                accuracies["epoch"] = epoch
                accuracies["bleu_score_train"] = np.mean(bleu_scores)
                accuracies["num_samples"] = epoch * len(train_dataset) + batch_idx * args.batch_size

                accuracies_over_time.append(accuracies)
                pd.DataFrame(accuracies_over_time).to_csv(
                    os.path.join(args.out_checkpoints_dir, args.model + "_accuracies.csv")
                )
                print(
                    f"Batch {batch_idx}: train loss: {np.mean(losses):.3f} | RL: {np.mean(losses_rl):.3f} |"
                    f" supervised: {np.mean(losses_supervised):.3f} | BLEU score (train): "
                    f"{np.mean(bleu_scores):.3f} | val loss: {val_loss:.3f} | captioning loss:"
                    f" {captioning_loss:.3f} | ranking loss: {ranking_loss:.3f} | val acc: {val_acc:.3f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(
                        model,
                        optimizer,
                        best_val_loss,
                        epoch,
                        os.path.join(args.out_checkpoints_dir, args.model + ".pt"),
                    )

            model.train()

            # Forward pass: RL
            loss, reward = forward_pass_rl(model, images, captions, vocab, args)
            losses_rl.append(loss.item())

            # Do another forward pass for supervised learning
            if args.weight_supervised_loss > 0:
                # Sample target sentences:
                sentence_idx = [random.choice(range(captions.shape[1])) for _ in range(images.shape[0])]
                target_captions = captions[torch.arange(captions.shape[0]), sentence_idx]
                target_caption_lengths = caption_lengths[torch.arange(captions.shape[0]), sentence_idx]
                target_captions = target_captions[:, :max(target_caption_lengths)]

                loss_supervised = forward_pass(model, images, target_captions, target_caption_lengths, args)
                losses_supervised.append(loss_supervised.item())

                loss += args.weight_supervised_loss * loss_supervised

            losses.append(loss.item())
            bleu_scores.append(reward.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"End of epoch: {epoch} | train loss: {np.mean(losses)} | BLEU score: {np.mean(bleu_scores):.3f} | "
            f"best val loss: {best_val_loss}\n\n"
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="show_attend_and_tell",
        choices=["show_and_tell", "show_attend_and_tell", "joint", "interactive"],
    )
    parser.add_argument(
        "--checkpoint", help="Path to checkpoint of pre-trained model",
    )
    parser.add_argument(
        "--out-checkpoints-dir",
        type=str,
        default=os.path.join(
            pathlib.Path.home(), "data/visual_ref/checkpoints/captioning_finetuned"
        ),
        help="Directory to which checkpoints should be saved to",
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
        "--n-epochs",
        type=int,
        default=15,
        help="Number of epochs to train (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--entropy-coeff", type=float, default=0.1, help="Entropy coefficient for RL",
    )
    parser.add_argument(
        "--length-cost", type=float, default=0.0, help="Length penalty for RL",
    )
    parser.add_argument(
        "--eval-semantics",
        default=False,
        action="store_true",
        help="Eval semantics of model using 2AFC",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--training-set-size",
        type=float,
        default=1.0,
        help="Training set size (as fraction of total data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--weight-supervised-loss",
        default=0,
        type=float,
        help="Supervised loss weight",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Start training on device: ", device)
    args = get_args()
    print(args)
    main(args)
