import argparse
import os
import pathlib
import pickle
import random
from typing import Optional, Dict

import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from torch.nn import functional as F

import matplotlib.pyplot as plt
from torchtext.vocab import Vocab

import egg.core as core
from egg.core import (
    Callback,
    Interaction,
    LoggingStrategy,
)

import pandas as pd

from dataset import VisualRefGameDataset, pad_collate_visual_ref
from eval_semantics import eval_semantics_score, get_semantics_eval_dataloader
from game import SenderReceiverRnnMultiTask
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker
from models.interactive.models import (
    VisualRefListenerOracle,
    VisualRefSpeakerDiscriminativeOracle,
    ImageEncoder,
    RnnSenderMultitaskVisualRef,
    loss_cross_entropy,
)
from preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    TOKEN_PADDING,
    TOKEN_START,
    TOKEN_END,
)
from trainers import VisualRefTrainer
from utils import (
    decode_caption,
    DEFAULT_WORD_EMBEDDINGS_SIZE,
    DEFAULT_LSTM_HIDDEN_SIZE,
    SEMANTICS_EVAL_FILES,
    CHECKPOINT_NAME_SENDER,
    DEFAULT_MAX_NUM_VAL_SAMPLES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrintDebugEvents(Callback):
    def __init__(
        self, vocab, train_dataset, val_dataset, semantics_eval_loaders, sender, args
    ):
        super().__init__()

        self.vocab = vocab

        self.train_loss = 0
        self.train_accuracies = 0
        self.train_func_loss = 0
        self.train_struct_loss = 0
        self.args = args

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.sender = sender
        self.semantics_eval_loaders = semantics_eval_loaders

        self.accuracies_over_time = []

    def print_interactions(self, interaction_logs, show_images, num_interactions=5):
        for _ in range(num_interactions):
            z = random.randint(0, interaction_logs.size - 1)
            message = decode_caption(interaction_logs.message[z], self.vocab)
            target_position = interaction_logs.labels[z]
            receiver_guess = torch.argmax(interaction_logs.receiver_output[z])
            target_image_id, distractor_image_id = interaction_logs.sender_input[z]

            if show_images:
                # plot the two images side-by-side
                target_image = self.train_dataset.get_image_features(
                    int(target_image_id), channels_first=False, normalize=False
                )
                distractor_image = self.train_dataset.get_image_features(
                    int(distractor_image_id), channels_first=False, normalize=False
                )
                image = torch.cat([target_image, distractor_image], dim=1).cpu().numpy()

                plt.title(
                    f"Left: Target, Right: Distractor"
                    f"\nReceiver guess correct: {target_position == receiver_guess}"
                    f"\nMessage: {message}"
                )
                plt.imshow(image)
                plt.show()
            else:
                print(
                    f"Target image ID: {target_image_id} | Distractor image ID: {distractor_image_id} | "
                    f"Success: {target_position == receiver_guess} | Message: {message}"
                )

    def on_test_end(self, loss, interaction_logs: Interaction, batch_id: int):
        loss_func = 0
        loss_struct = 0
        val_acc = interaction_logs.aux["acc"].mean().item()
        if "loss_functional" in interaction_logs.aux:
            loss_func = interaction_logs.aux["loss_functional"].mean().item()
        if "loss_structural" in interaction_logs.aux:
            loss_struct = interaction_logs.aux["loss_structural"].mean().item()
        accuracies = {
            "batch_id": batch_id,
            "val_loss": loss,
            "val_acc": val_acc,
            "val_loss_func": loss_func,
            "val_loss_struct": loss_struct,
        }
        print(
            f"EVAL Batch {batch_id}: loss: {loss:.3f} loss_func: {loss_func:.3f} loss_struct: {loss_struct:.3f} "
            f"accuracy: {val_acc:.3f}"
        )

        if self.args.eval_semantics:
            for name, semantic_images_loader in self.semantics_eval_loaders.items():
                acc = eval_semantics_score(
                    self.sender, semantic_images_loader, self.vocab
                )
                print(f"Accuracy for {name}: {acc:.3f}")
                accuracies[name] = acc

        self.accuracies_over_time.append(accuracies)
        pd.DataFrame(self.accuracies_over_time).to_csv(
            os.path.join(
                args.out_checkpoints_dir,
                CHECKPOINT_NAME_SENDER.replace(".pt", "_accuracies.csv"),
            )
        )

        if args.print_sample_interactions or args.print_sample_interactions_images:
            self.print_interactions(
                interaction_logs, show_images=args.print_sample_interactions_images,
            )

    def on_batch_end(
        self,
        interaction_logs: Interaction,
        loss: float,
        batch_id: int,
        is_training: bool = True,
    ):
        if is_training:
            if batch_id == 0:
                self.train_loss = 0
                self.train_accuracies = 0
                self.train_func_loss = 0
                self.train_struct_loss = 0

            self.train_loss += loss
            if "loss_functional" in interaction_logs.aux:
                self.train_accuracies += interaction_logs.aux["acc"].sum()
                self.train_func_loss += interaction_logs.aux["loss_functional"].item()
            if "loss_structural" in interaction_logs.aux:
                self.train_struct_loss += interaction_logs.aux["loss_structural"].item()

            if (batch_id + 1) % self.args.log_frequency == 0:
                loss = self.train_loss / self.args.log_frequency
                batch_size = interaction_logs.labels.shape[0]
                mean_acc = self.train_accuracies / (
                    self.args.log_frequency * batch_size
                )
                loss_func = self.train_func_loss / self.args.log_frequency
                loss_struct = self.train_struct_loss / self.args.log_frequency

                print(
                    f"Batch {batch_id + 1}: loss: {loss:.3f} loss_func: {loss_func:.3f} loss_struct: {loss_struct:.3f} "
                    f"accuracy: {mean_acc:.3f}"
                )

                self.train_loss = 0
                self.train_accuracies = 0
                self.train_func_loss = 0
                self.train_struct_loss = 0


def loss_functional(
    sender_input, message, sender_logits, receiver_input, receiver_output, labels
):
    # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
    # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
    # loss = F.cross_entropy(receiver_output, labels, reduction="none")
    loss = - acc
    return loss, {"acc": acc}


def loss_structural(
    captions, sender_logits,
):
    loss = loss_cross_entropy(sender_logits, captions)

    return loss, None


def main(args):
    # create model checkpoint directory
    if not os.path.exists(args.out_checkpoints_dir):
        os.makedirs(args.out_checkpoints_dir)

    train_dataset = VisualRefGameDataset(
        DATA_PATH, IMAGES_FILENAME["train"], CAPTIONS_FILENAME["train"], args.batch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate_visual_ref,
    )
    val_dataset = VisualRefGameDataset(
        DATA_PATH,
        IMAGES_FILENAME["val"],
        CAPTIONS_FILENAME["val"],
        args.batch_size,
        max_samples=args.max_val_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate_visual_ref,
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    sender_vocab = vocab
    if args.sender_vocab_size:
        # create a new vocab with smaller number of elements
        sender_vocab = Vocab(
            vocab.freqs,
            max_size=args.sender_vocab_size,
            specials=[TOKEN_PADDING, Vocab.UNK, TOKEN_START, TOKEN_END],
        )

    semantics_eval_loaders = {
        file: get_semantics_eval_dataloader(file, vocab)
        for file in SEMANTICS_EVAL_FILES
    }

    word_embedding_size = DEFAULT_WORD_EMBEDDINGS_SIZE

    args.sender_hidden = DEFAULT_LSTM_HIDDEN_SIZE
    args.sender_embedding = word_embedding_size
    args.receiver_embedding = word_embedding_size  # ???
    args.receiver_hidden = DEFAULT_LSTM_HIDDEN_SIZE  # ???
    args.sender_cell = "lstm"
    args.receiver_cell = "lstm"

    joint_embeddings_size = DEFAULT_LSTM_HIDDEN_SIZE
    lstm_hidden_size = DEFAULT_LSTM_HIDDEN_SIZE

    if args.sender == "oracle":
        sender = VisualRefSpeakerDiscriminativeOracle(
            DATA_PATH, CAPTIONS_FILENAME, args.max_len, vocab
        )

    elif args.sender == "functional":
        encoder = ImageEncoder(joint_embeddings_size, fine_tune_resnet=False)
        sender = RnnSenderMultitaskVisualRef(
            encoder,
            vocab=sender_vocab,
            embed_dim=args.sender_embedding,
            hidden_size=args.sender_hidden,
            cell=args.sender_cell,
            max_len=args.max_len,
        )

        if args.sender_checkpoint:
            sender_checkpoint = torch.load(args.sender_checkpoint, map_location=device)
            sender.load_state_dict(sender_checkpoint["model_state_dict"])

    else:
        raise ValueError(f"Unknown sender model type: {args.sender}")

    # use custom LoggingStrategy that stores image IDs
    logging_strategy = VisualRefLoggingStrategy()

    receivers = []
    for checkpoint in args.receiver_checkpoints:
        checkpoint_ranking_model = torch.load(checkpoint, map_location=device)
        ranking_model = ImageSentenceRanker(
            word_embedding_size,
            joint_embeddings_size,
            lstm_hidden_size,
            len(vocab),
            fine_tune_resnet=False,
        )
        ranking_model.load_state_dict(checkpoint_ranking_model["model_state_dict"])
        ranking_model = ranking_model.to(device)
        receiver = VisualRefListenerOracle(ranking_model)

        if args.freeze_receiver:
            for param in receiver.parameters():
                param.requires_grad = False

        receiver = receiver.to(device)
        receivers.append(receiver)

    game = SenderReceiverRnnMultiTask(
        sender,
        receivers,
        loss_functional=loss_functional,
        loss_structural=loss_structural,
        sender_entropy_coeff=args.sender_entropy_coeff,
        length_cost=args.length_cost,
        receiver_entropy_coeff=0,
        train_logging_strategy=logging_strategy,
        test_logging_strategy=logging_strategy,
        weight_structural_loss=args.weight_structural_loss,
        weight_functional_loss=args.weight_functional_loss,

    )

    callbacks = [
        PrintDebugEvents(
            vocab, train_dataset, val_dataset, semantics_eval_loaders, sender, args
        ),
    ]

    optimizer = core.build_optimizer(game.parameters())

    trainer = VisualRefTrainer(
        game=game,
        optimizer=optimizer,
        out_checkpoints_dir=args.out_checkpoints_dir,
        train_data=train_loader,
        vocab=vocab,
        validation_data=val_loader,
        callbacks=callbacks,
    )

    print("Starting training with args: ")
    print(args)
    print("Number of samples: ", len(train_dataset))
    trainer.train(args.n_epochs)

    game.eval()

    core.close()


@dataclass
class VisualRefLoggingStrategy(LoggingStrategy):
    def filtered_interaction(
        self,
        sender_input: Optional[torch.Tensor],
        receiver_input: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        message: Optional[torch.Tensor],
        receiver_output: Optional[torch.Tensor],
        message_length: Optional[torch.Tensor],
        aux: Dict[str, torch.Tensor],
    ):
        # Store only image IDs but not data
        (
            target_image,
            distractor_image,
            target_image_id,
            distractor_image_id,
            captions,
            sequence_lengths,
        ) = sender_input

        filtered_sender_input = torch.tensor(
            list(zip(target_image_id, distractor_image_id))
        )

        return Interaction(
            sender_input=filtered_sender_input,
            receiver_input=None,
            labels=labels,
            message=message,
            receiver_output=receiver_output,
            message_length=message_length,
            aux=aux,
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-sample-interactions",
        default=False,
        action="store_true",
        help="Print sample interactions output.",
    )
    parser.add_argument(
        "--print-sample-interactions-images",
        default=False,
        action="store_true",
        help="Print sample interactions output with images.",
    )
    parser.add_argument(
        "--log-frequency",
        default=100,
        type=int,
        help="Logging frequency (number of batches)",
    )
    parser.add_argument(
        "--eval-frequency",
        default=100,
        type=int,
        help="Evaluation frequency (number of batches)",
    )
    parser.add_argument(
        "--sender",
        default="oracle",
        type=str,
        choices=["oracle", "functional"],
        help="Sender model",
    )
    parser.add_argument(
        "--sender-entropy-coeff",
        default=0,
        type=float,
        help="Sender entropy regularization coefficient",
    )
    parser.add_argument(
        "--length-cost",
        default=0.0,
        type=float,
        help="penalty applied to Sender for each symbol produced",
    )
    parser.add_argument(
        "--sender-checkpoint",
        type=str,
        help="Checkpoint to load the sender model from",
    )
    parser.add_argument(
        "--receiver-checkpoints",
        type=str,
        nargs="+",
        help="Checkpoint to load the receiver models from",
    )
    parser.add_argument(
        "--out-checkpoints-dir",
        type=str,
        default=os.path.join(
            pathlib.Path.home(), "data/visual_ref/checkpoints/interactive"
        ),
        help="Directory to which checkpoints should be saved to",
    )
    parser.add_argument(
        "--max-val-samples",
        default=DEFAULT_MAX_NUM_VAL_SAMPLES,
        type=int,
        help="Maximum number of samples for validation",
    )
    parser.add_argument(
        "--freeze-receiver",
        default=False,
        action="store_true",
        help="Freeze receiver weights",
    )
    parser.add_argument(
        "--eval-semantics",
        default=False,
        action="store_true",
        help="Eval semantics of model using 2AFC",
    )
    parser.add_argument(
        "--weight-structural-loss",
        default=1.0,
        type=float,
        help="Structural loss weight",
    )
    parser.add_argument(
        "--weight-functional-loss",
        default=1.0,
        type=float,
        help="Functional loss weight",
    )
    parser.add_argument(
        "--sender-vocab-size",
        default=None,
        type=int,
        help="Max size of the sender vocab",
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
