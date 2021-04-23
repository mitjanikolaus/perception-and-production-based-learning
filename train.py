import argparse
import os
import pathlib
import pickle
from typing import Optional, Dict

import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from torch.nn import functional as F

import egg.core as core
from egg.core import (
    ConsoleLogger,
    Callback,
    Interaction,
    SenderReceiverRnnReinforce,
    LoggingStrategy,
)
from dataset import VisualRefGameDataset, pad_collate_visual_ref
from eval_semantics import eval_semantics_score, get_semantics_eval_dataloader
from game import SenderReceiverRnnMultiTask
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker
from models.interactive.models import (
    VisualRefListenerOracle,
    VisualRefSpeakerDiscriminativeOracle,
    ImageEncoder,
    RnnSenderMultitaskVisualRef, loss_cross_entropy,
)
from preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
)
from trainers import VisualRefTrainer
from utils import decode_caption, DEFAULT_WORD_EMBEDDINGS_SIZE, DEFAULT_LSTM_HIDDEN_SIZE, SEMANTICS_EVAL_FILES, \
    CHECKPOINT_NAME_SENDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_NUM_VAL_SAMPLES = 3200


class PrintDebugEvents(Callback):
    def __init__(self, vocab, train_dataset, val_dataset, args):
        super().__init__()

        self.vocab = vocab

        self.train_loss = 0
        self.train_accuracies = 0
        self.args = args

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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


def loss_functional(
    sender_input, message, sender_logits, receiver_input, receiver_output, labels
):
    # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
    # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def loss_structural(
    sender_input, message, sender_logits, receiver_input, receiver_output, labels
):
    (
        images,
        target_label,
        target_image_ids,
        distractor_image_ids,
        captions,
        sequence_lengths,
    ) = sender_input

    loss = loss_cross_entropy(sender_logits, captions)

    return loss, None


# def loss_multitask(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels, weight_structural=1.0):
#     loss_str, _ = loss_structural(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels)
#     loss_func, acc = loss_functional(_sender_input, _message, sender_logits, _receiver_input, receiver_output, labels)
#     loss_func = loss_func.mean()
#     print(f"Structural Loss: {loss_str:.3f} Functional Loss: {loss_func:.3f}")
#     return weight_structural * loss_str + loss_func, acc


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
        max_samples=MAX_NUM_VAL_SAMPLES,
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
    args.max_len = 25
    args.random_seed = 1

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
    checkpoint_ranking_model = torch.load(args.receiver_checkpoint, map_location=device)
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
        encoder = ImageEncoder(
            joint_embeddings_size, fine_tune_resnet=False
        )
        sender = RnnSenderMultitaskVisualRef(
            encoder,
            vocab_size=len(vocab),
            vocab=vocab,
            embed_dim=args.sender_embedding,
            hidden_size=args.sender_hidden,
            cell=args.sender_cell,
            max_len=args.max_len,
        )

        if args.sender_checkpoint:
            sender_checkpoint = torch.load(args.sender_checkpoint, map_location=device)
            sender.load_state_dict(sender_checkpoint['model_state_dict'])

    else:
        raise ValueError(f"Unknown sender model type: {args.sender}")

    # use custom LoggingStrategy that stores image IDs
    logging_strategy = VisualRefLoggingStrategy()

    game = SenderReceiverRnnMultiTask(
        sender,
        receiver,
        loss_functional=loss_functional,
        loss_structural=loss_structural,
        sender_entropy_coeff=args.sender_entropy_coeff,
        length_cost=args.length_cost,
        receiver_entropy_coeff=0,
        train_logging_strategy=logging_strategy,
        test_logging_strategy=logging_strategy,
    )

    callbacks = [ConsoleLogger(print_train_loss=True, as_json=False),
                 PrintDebugEvents(vocab, train_dataset, val_dataset, args)]

    optimizer = core.build_optimizer(game.parameters())

    trainer = VisualRefTrainer(
        game=game,
        optimizer=optimizer,
        out_checkpoints_dir=args.out_checkpoints_dir,
        train_data=train_loader,
        semantics_eval_loaders=semantics_eval_loaders,
        vocab=vocab,
        validation_data=val_loader,
        callbacks=callbacks,
        print_sample_interactions=args.print_sample_interactions,
        print_sample_interactions_images=args.print_sample_interactions_images,
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
        "--sender-checkpoint",
        type=str,
        help="Checkpoint to load the sender model from",
    )
    parser.add_argument(
        "--receiver-checkpoint",
        type=str,
        help="Checkpoint to load the receiver model from",
    )
    parser.add_argument(
        "--out-checkpoints-dir",
        type=str,
        default=os.path.join(pathlib.Path.home(), "data/visual_ref/checkpoints/interactive"),
        help="Directory to which checkpoints should be saved to",
    )
    parser.add_argument(
        "--max-val-samples",
        default=MAX_NUM_VAL_SAMPLES,
        type=int,
        help="Maximum number of samples for validation",
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
