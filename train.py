import argparse
import os
import pickle
from typing import Optional, Dict

import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from torch.nn import functional as F

import matplotlib.pyplot as plt

import egg.core as core
from egg.core import ConsoleLogger, Callback, Interaction, SenderReceiverRnnReinforce, LoggingStrategy
from dataset import VisualRefGameDataset, pad_collate_visua_ref
from game import SenderReceiverRnnMultiTask
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker
from models.interactive.models import VisualRefListenerOracle, VisualRefSpeakerDiscriminativeOracle, \
    VisualRefSenderFunctional, RnnSenderReinforceVisualRef
from preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
)
from trainers import VisualRefTrainer
from utils import decode_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrintDebugEvents(Callback):
    def __init__(self, train_dataset, val_dataset, args):
        super().__init__()

        vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
        with open(vocab_path, "rb") as file:
            self.vocab = pickle.load(file)

        self.train_loss = 0
        self.train_accuracies = 0
        self.args = args

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def print_sample_interactions(
        self, interaction_logs, show_images, num_interactions=5
    ):
        for z in range(num_interactions):
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

    def on_test_end(self, _loss, interaction_logs: Interaction, epoch: int):
        pass

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

            self.train_loss += loss.detach()
            self.train_accuracies += interaction_logs.aux["acc"].sum()

            if (batch_id % self.args.log_frequency) == (self.args.log_frequency - 1):
                mean_loss = self.train_loss / self.args.log_frequency
                batch_size = interaction_logs.aux["acc"].size()[0]
                mean_acc = self.train_accuracies / (
                    self.args.log_frequency * batch_size
                )

                print(
                    f"Batch {batch_id + 1}: loss: {mean_loss:.3f} accuracy: {mean_acc:.3f}"
                )

                self.train_loss = 0
                self.train_accuracies = 0

                if (
                    self.args.print_sample_interactions
                    or self.args.print_sample_interactions_images
                ):
                    self.print_sample_interactions(
                        interaction_logs,
                        show_images=self.args.print_sample_interactions_images,
                    )


def loss_functional(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels):
    # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
    # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def loss_structural(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels):
    images, target_label, target_image_ids, distractor_image_ids, captions, sequence_lengths = _sender_input

    # trim logits to max target caption length
    sender_logits = sender_logits[:captions.size(1)]
    sender_logits = torch.stack(sender_logits).permute(1, 2, 0)

    # use NLL loss as logits are already softmaxed
    loss = F.nll_loss(sender_logits, captions, ignore_index=0)

    return loss, None


def loss_multitask(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels, weight_structural=1.0):
    loss_str, _ = loss_structural(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels)
    loss_func, acc = loss_functional(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels)
    loss_func = loss_func.mean()
    print(f"Structural Loss: {loss_str:.3f} Functional Loss: {loss_func:.3f}")
    return weight_structural * loss_str + loss_func, acc


def main(args):
    train_dataset = VisualRefGameDataset(
        DATA_PATH, IMAGES_FILENAME["train"], CAPTIONS_FILENAME["train"], args.batch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate_visua_ref
    )
    val_dataset = VisualRefGameDataset(
        DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"], args.batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate_visua_ref
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    # TODO
    # TODO: embedding size for speaker is 1024 in paper
    args.sender_hidden = 512  # TODO
    args.sender_embedding = 512  # ???
    args.receiver_embedding = 100  # ???
    args.receiver_hidden = 512  # ???
    args.sender_cell = "lstm"
    args.receiver_cell = "lstm"
    args.max_len = 25
    args.random_seed = 1

    word_embedding_size = 100
    joint_embeddings_size = 512
    lstm_hidden_size = 512

    # if args.receiver_checkpoint:
    #     checkpoint_listener = torch.load(args.receiver_checkpoint, map_location=device)
    #     ranking_model = ImageSentenceRanker(
    #         word_embedding_size,
    #         joint_embeddings_size,
    #         lstm_hidden_size,
    #         len(vocab),
    #         fine_tune_resnet=False,
    #     )
    #     receiver = VisualRefListenerOracle(ranking_model)
    #     receiver.load_state_dict(checkpoint_listener["model_state_dict"])
    # else:
    checkpoint_ranking_model = torch.load(
        args.receiver_checkpoint, map_location=device
    )
    ranking_model = ImageSentenceRanker(
        word_embedding_size,
        joint_embeddings_size,
        lstm_hidden_size,
        len(vocab),
        fine_tune_resnet=False,
    )
    ranking_model.load_state_dict(checkpoint_ranking_model["model_state_dict"])
    receiver = VisualRefListenerOracle(ranking_model)

    if args.freeze_receiver:
        for param in receiver.parameters():
            param.requires_grad = False

    if args.sender == "oracle":
        sender = VisualRefSpeakerDiscriminativeOracle(
            DATA_PATH, CAPTIONS_FILENAME, args.max_len, vocab
        )

    elif args.sender == "functional":
        sender = VisualRefSenderFunctional(
            joint_embeddings_size, fine_tune_resnet=False
        )
        sender = RnnSenderReinforceVisualRef(
            sender,
            vocab_size=len(vocab),
            embed_dim=args.sender_embedding,
            hidden_size=args.sender_hidden,
            cell=args.sender_cell,
            max_len=args.max_len,
        )

    else:
        raise ValueError(f"Unknown sender model type: {args.sender}")

    # use custom LoggingStrategy that stores image IDs
    logging_strategy = VisualRefLoggingStrategy()

    game = SenderReceiverRnnMultiTask(
        sender,
        receiver,
        loss_multitask,
        sender_entropy_coeff=args.sender_entropy_coeff,
        length_cost=args.length_cost,
        receiver_entropy_coeff=0,
        train_logging_strategy=logging_strategy,
        test_logging_strategy=logging_strategy,
    )

    callbacks = [ConsoleLogger(print_train_loss=True, as_json=False)]
    # core.PrintValidationEvents(n_epochs=1)
    callbacks.append(PrintDebugEvents(train_dataset, val_dataset, args))

    optimizer = core.build_optimizer(game.parameters())

    trainer = VisualRefTrainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
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
            sequence_lengths
        ) = sender_input


        filtered_sender_input = list(zip(target_image_id, distractor_image_id))

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
        default=0.1,
        type=float,
        help="Sender entropy coefficient",
    )
    parser.add_argument(
        "--length-cost",
        default=0.0,
        type=float,
        help="penalty applied to Sender for each symbol produced",
    )
    parser.add_argument(
        "--receiver-checkpoint",
        type=str,
        help="Checkpoint to load the receiver model from",
    )
    parser.add_argument(
        "--freeze-receiver",
        default=False,
        action="store_true",
        help="Freeze receiver weights",
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
