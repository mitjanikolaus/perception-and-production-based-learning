import argparse
import pathlib
import pickle
import os
import random

import numpy as np

import pandas as pd

import torch
import torch.distributions
import torch.utils.data
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CaptionRLDataset
from eval_semantics import eval_semantics_score, get_semantics_eval_dataloader
from generate_semantics_eval_dataset import VERBS
from models.image_captioning.show_and_tell import ShowAndTell
from models.image_captioning.show_attend_and_tell import ShowAttendAndTell
from models.image_sentence_ranking.ranking_model import accuracy_discrimination
from models.interactive.models import loss_cross_entropy
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
    SEMANTICS_EVAL_FILES,
    DEFAULT_LOG_FREQUENCY,
    DEFAULT_WORD_EMBEDDINGS_SIZE,
    DEFAULT_LSTM_HIDDEN_SIZE,
    LEGEND,
    set_seeds,
    decode_caption,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRINT_SAMPLE_CAPTIONS = 5

NUM_BATCHES_VALIDATION = 100

NUM_VALIDATIONS_NO_IMPROVEMENT_EARLY_STOPPING = 50

WEIGH_RANKING_LOSS = 1

UNIQUE_VERBS = list(np.unique(np.array(VERBS).flatten()))


def print_model_output(output, target_captions, image_ids, vocab, num_captions=1):
    captions_model = torch.argmax(output, dim=1)
    print_captions(captions_model, target_captions, image_ids, vocab, num_captions)


def print_captions(captions, target_captions, image_ids, vocab, num_captions=1):
    for i in range(num_captions):
        print(f"Image ID: {image_ids[i]}")
        print("Target: ", end="")
        print_caption(target_captions[i][0], vocab)
        print("Model output: ", end="")
        print_caption(captions[i], vocab)


def print_sample_model_output(model, dataloader, vocab, num_captions=5):
    images, target_captions, caption_lengths, image_ids = next(iter(dataloader))

    captions, _, _, _ = model.decode(images)

    print_captions(captions, target_captions, image_ids, vocab, num_captions)


def print_produced_utterances_stats(produced_utterances):
    mean_seq_length = np.mean([len(sequence.split(' ')) for sequence in produced_utterances])
    jenny_occurrences = np.mean(['jenny' in sequence.split(' ') and not 'mike' in sequence.split(' ') for sequence in produced_utterances])
    mike_occurrences = np.mean(['mike' in sequence.split(' ') and not 'jenny' in sequence.split(' ') for sequence in produced_utterances])
    print(f"Mean seq length: {mean_seq_length:.3f}")
    print(f"Seq containing 'jenny' (and not 'mike'): {jenny_occurrences:.3f}")
    print(f"Seq containing 'mike' (and not 'jenny'): {mike_occurrences:.3f}")

    stats = {
        "seq_lengths": mean_seq_length,
        "jenny_occurrences": jenny_occurrences,
        "mike_occurrences": mike_occurrences,
    }

    for verb in UNIQUE_VERBS:
        verb_occurrences = np.mean([verb in sequence.split(' ') for sequence in produced_utterances])
        print(f"Seq containing '{verb}': {verb_occurrences:.3f}")
        stats[verb] = verb_occurrences

    return stats



def validate_model(
    model,
    dataloader,
    semantic_images_loaders,
    vocab,
    args,
    val_bleu_score=False,
    max_batches=NUM_BATCHES_VALIDATION,
    return_produced_sequences=False,
):
    semantic_accuracies = {}

    model.eval()
    with torch.no_grad():
        print_sample_model_output(model, dataloader, vocab, PRINT_SAMPLE_CAPTIONS)
        if args.eval_semantics:
            for name, semantic_images_loader in semantic_images_loaders.items():
                acc = eval_semantics_score(model, semantic_images_loader, vocab)
                print(f"Accuracy for {LEGEND[name]}: {acc:.3f}")
                semantic_accuracies[name] = acc

        val_losses = []
        captioning_losses = []
        ranking_losses = []
        val_accuracies = []
        bleu_scores = []
        produced_sequences = []

        for batch_idx, (images, captions, caption_lengths, _) in enumerate(dataloader):
            if val_bleu_score:
                # Validate by calculating BLEU score
                sequences, logits, entropies, sequence_lengths = model.decode(
                    images, sampling=True
                )

                bleu_scores_batch = model.reward_rl(
                    sequences, captions, vocab, args.weights_bleu,
                )
                bleu_scores.extend(bleu_scores_batch.tolist())

                if return_produced_sequences:
                    produced_sequences.extend([
                        decode_caption(sequence, vocab) for sequence in sequences
                    ])

            else:
                if args.model == "joint":
                    (
                        scores,
                        decode_lengths,
                        alphas,
                        images_embedded,
                        captions_embedded,
                    ) = model(images, captions, caption_lengths)
                    loss_captioning, loss_ranking = model.loss(
                        scores,
                        captions,
                        decode_lengths,
                        alphas,
                        images_embedded,
                        captions_embedded,
                    )

                    # TODO weigh losses appropriately
                    loss_ranking = WEIGH_RANKING_LOSS * loss_ranking
                    loss = loss_captioning + loss_ranking

                    acc = accuracy_discrimination(images_embedded, captions_embedded)
                    val_accuracies.append(acc)

                    captioning_losses.append(loss_captioning.item())
                    ranking_losses.append(loss_ranking.item())
                elif args.model == "interactive":
                    scores, decode_lengths, _ = model(images, captions, caption_lengths)
                    loss = loss_cross_entropy(scores, captions)

                else:
                    scores, decode_lengths, alphas = model(
                        images, captions, caption_lengths
                    )
                    loss = model.loss(scores, captions, decode_lengths, alphas)

                val_losses.append(loss.mean().item())

            if max_batches and batch_idx > max_batches:
                break

    model.train()
    return (
        np.mean(val_losses),
        semantic_accuracies,
        np.mean(captioning_losses),
        np.mean(ranking_losses),
        np.mean(val_accuracies),
        np.mean(bleu_scores),
        produced_sequences,
    )


def save_model(model, optimizer, best_bleu_score, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "BLEU": best_bleu_score,
        },
        path,
    )


def forward_pass_supervised(model, images, captions, caption_lengths, args):
    model.train()

    # Forward pass
    if args.model == "joint":
        (scores, decode_lengths, alphas, images_embedded, captions_embedded,) = model(
            images, captions, caption_lengths
        )
        loss_captioning, loss_ranking = model.loss(
            scores,
            captions,
            decode_lengths,
            alphas,
            images_embedded,
            captions_embedded,
        )
        # TODO weigh losses appropriately
        loss_ranking = WEIGH_RANKING_LOSS * loss_ranking
        loss = loss_captioning + loss_ranking
    elif args.model == "interactive":
        scores, _, _ = model(images, captions, caption_lengths)
        loss = loss_cross_entropy(scores, captions)
    else:
        scores, decode_lengths, alphas = model(images, captions, caption_lengths)
        loss = model.loss(scores, captions, decode_lengths, alphas)

    return loss.mean()


def forward_pass_rl(model, images, captions, vocab, args):
    sequences, logits, entropies, sequence_lengths = model.decode(images, sampling=True)

    reward = model.reward_rl(sequences, captions, vocab, args.weights_bleu,)

    # # the log prob/ entropy of the choices made by S before and including the eos symbol
    effective_entropy = entropies.sum(dim=1) / sequence_lengths.float()
    effective_log_prob = logits.sum(dim=1) / sequence_lengths.float()

    entropy_loss = effective_entropy.mean() * args.entropy_coeff

    length_loss = sequence_lengths.float() * args.length_cost

    policy_length_loss = (length_loss * effective_log_prob).mean()
    policy_loss = -(reward * effective_log_prob).mean()

    rl_loss = policy_length_loss + policy_loss - entropy_loss

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
        DATA_PATH,
        IMAGES_FILENAME["train"],
        CAPTIONS_FILENAME["train"],
        vocab,
        args.training_set_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionRLDataset.pad_collate,
    )
    val_images_loader = DataLoader(
        CaptionRLDataset(
            DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"], vocab
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=CaptionRLDataset.pad_collate,
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

    best_bleu_score = 0
    validations_no_improvement = 0
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
                    val_bleu_score,
                    produced_utterances,
                ) = validate_model(
                    model,
                    val_images_loader,
                    semantics_eval_loaders,
                    vocab,
                    args,
                    val_bleu_score=True,
                    return_produced_sequences=args.log_produced_utterances_stats,
                )
                accuracies["val_loss"] = val_loss
                accuracies["bleu_score_val"] = val_bleu_score
                accuracies["batch_id"] = batch_idx
                accuracies["epoch"] = epoch
                accuracies["bleu_score_train"] = np.mean(bleu_scores)
                accuracies["num_samples"] = (
                    epoch * len(train_dataset) + batch_idx * args.batch_size
                )
                if args.log_produced_utterances_stats:
                    stats = print_produced_utterances_stats(produced_utterances)
                    accuracies.update(stats)

                accuracies_over_time.append(accuracies)
                pd.DataFrame(accuracies_over_time).to_csv(
                    os.path.join(
                        args.out_checkpoints_dir,
                        f"{args.model}_train_frac_{args.training_set_size}_accuracies.csv",
                    )
                )
                print(
                    f"Batch {batch_idx}: train loss: {np.mean(losses):.3f} | RL: {np.mean(losses_rl):.3f} |"
                    f" supervised: {np.mean(losses_supervised):.3f} | BLEU score (train): "
                    f"{np.mean(bleu_scores):.3f} | BLEU score (val): "
                    f"{val_bleu_score:.3f} | val loss: {val_loss:.3f} | captioning loss:"
                    f" {captioning_loss:.3f} | ranking loss: {ranking_loss:.3f} | val acc: {val_acc:.3f}"
                )

                if val_bleu_score > best_bleu_score:
                    best_bleu_score = val_bleu_score
                    save_model(
                        model,
                        optimizer,
                        best_bleu_score,
                        epoch,
                        os.path.join(
                            args.out_checkpoints_dir,
                            f"{args.model}_train_frac_{args.training_set_size}.pt",
                        ),
                    )
                    validations_no_improvement = 0
                else:
                    validations_no_improvement += 1
                    if (
                        validations_no_improvement
                        >= NUM_VALIDATIONS_NO_IMPROVEMENT_EARLY_STOPPING
                    ):
                        print(
                            f"\nEarly stopping: no improvement for {validations_no_improvement} validations"
                        )
                        return

            model.train()

            # Alternative between supervised and RL
            if (
                args.frequency_rl_updates != -1
                and batch_idx % (args.frequency_rl_updates + 1) == 0
            ):
                #  Forward pass: Supervised

                # Sample target sentences:
                sentence_idx = [
                    random.choice(range(captions.shape[1]))
                    for _ in range(images.shape[0])
                ]
                target_captions = captions[
                    torch.arange(captions.shape[0]), sentence_idx
                ]
                target_caption_lengths = caption_lengths[
                    torch.arange(captions.shape[0]), sentence_idx
                ]
                target_captions = target_captions[:, : max(target_caption_lengths)]

                loss_supervised = forward_pass_supervised(
                    model, images, target_captions, target_caption_lengths, args
                )
                losses_supervised.append(loss_supervised.item())

                loss = loss_supervised
            else:
                # Forward pass: RL
                loss, reward = forward_pass_rl(model, images, captions, vocab, args)
                losses_rl.append(loss.item())

                bleu_scores.append(reward.item())

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"End of epoch: {epoch} | train loss: {np.mean(losses)} | BLEU score (train): {np.mean(bleu_scores):.3f} | "
            f"best BLEU score (val): {best_bleu_score}\n\n"
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="show_and_tell",
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
        default=100,
        help="Number of epochs to train (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--entropy-coeff", type=float, default=0.0, help="Entropy coefficient for RL",
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
        "--seed", type=int, help="Random seed",
    )
    parser.add_argument(
        "--frequency-rl-updates",
        default=0,
        type=int,
        help="RL updates frequency (number of RL updates per supervised update, set to -1 for only RL, or to 0 for only"
        "supervised updates)",
    )
    parser.add_argument(
        "--weights-bleu",
        default=(0.25, 0.25, 0.25, 0.25),
        type=float,
        nargs=4,
        help="Weights for BLEU score that is used as reward for RL",
    )
    parser.add_argument(
        "--log-produced-utterances-stats", default=False, action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Start training on device: ", device)
    args = get_args()
    print(args)
    main(args)
