import random
from collections import defaultdict
from typing import Callable, Iterable

import torch
from torch import nn

from egg.core import LoggingStrategy, find_lengths
from egg.core.baselines import Baseline, MeanBaseline
from egg.core.reinforce_wrappers import _verify_batch_sizes
from utils import sequences, entropy


class SenderReceiverRnnMultiTask(nn.Module):
    """

    """

    def __init__(
        self,
        sender: nn.Module,
        receivers: Iterable[nn.Module],
        loss_functional: Callable,
        loss_structural: Callable,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        length_cost: float = 0.0,
        weight_structural_loss: float = 1.0,
        weight_functional_loss: float = 1.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks
        """
        super(SenderReceiverRnnMultiTask, self).__init__()
        self.sender = sender
        self.receivers = receivers
        self.loss_functional = loss_functional
        self.loss_structural = loss_structural

        self.mechanics = CommunicationRnnMultiTask(
            sender_entropy_coeff,
            receiver_entropy_coeff,
            length_cost,
            weight_structural_loss,
            weight_functional_loss,
            baseline_type,
            train_logging_strategy,
            test_logging_strategy,
        )

    def forward(self, sender_input, labels, receiver_input=None):
        return self.mechanics(
            self.sender,
            self.receivers,
            self.loss_functional,
            self.loss_structural,
            sender_input,
            labels,
            receiver_input,
        )


class CommunicationRnnMultiTask(nn.Module):
    def __init__(
        self,
        sender_entropy_coeff: float,
        receiver_entropy_coeff: float,
        length_cost: float = 0.0,
        weight_structural_loss: float = 1.0,
        weight_functional_loss: float = 1.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        """
        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks

        """
        super().__init__()

        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.length_cost = length_cost
        self.weight_structural_loss = weight_structural_loss
        self.weight_functional_loss = weight_functional_loss

        self.baselines = defaultdict(baseline_type)
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(
        self,
        sender,
        receivers,
        loss_functional,
        loss_structural,
        sender_input,
        labels,
        receiver_input=None,
    ):
        # Sample a random receiver
        receiver = random.choice(receivers)

        (
            images,
            target_label,
            target_image_ids,
            distractor_image_ids,
            captions,
            sequence_lengths,
        ) = sender_input
        images_target = images[target_label, range(images.size(1))]

        optimized_loss = 0
        aux_info = {}
        messages = None
        message_lengths = None
        receiver_output = None

        if self.weight_functional_loss > 0:
            # Forward pass without teacher forcing for RL loss
            if self.training:
                messages, log_prob_s, entropy_s, _ = sender(
                    images_target, use_teacher_forcing=False, decode_sampling=True,
                )
            else:
                messages, log_prob_s, entropy_s, _ = sender(
                    images_target, use_teacher_forcing=False, decode_sampling=True,
                )
            message_lengths = find_lengths(messages)

            receiver_output, log_prob_r, entropy_r = receiver(
                messages, receiver_input, message_lengths
            )

            loss, aux_info = loss_functional(
                sender_input, messages, log_prob_s, receiver_input, receiver_output, labels
            )

            # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
            effective_entropy_s = torch.zeros_like(entropy_r)
            #
            # # the log prob of the choices made by S before and including the eos symbol - again, we don't
            # # care about the rest
            effective_log_prob_s = torch.zeros_like(log_prob_r)
            #
            for i in range(max(message_lengths)):
                not_eosed = (i < message_lengths).float()
                effective_entropy_s += entropy_s[:, i] * not_eosed
                effective_log_prob_s += log_prob_s[:, i] * not_eosed
            effective_entropy_s = effective_entropy_s / message_lengths.float()

            weighted_entropy = (
                effective_entropy_s.mean() * self.sender_entropy_coeff
                + entropy_r.mean() * self.receiver_entropy_coeff
            )

            log_prob = effective_log_prob_s + log_prob_r

            length_loss = message_lengths.float() * self.length_cost

            policy_length_loss = (
                (length_loss - self.baselines["length"].predict(length_loss))
                * effective_log_prob_s
            ).mean()
            policy_loss = (
                (loss.detach() - self.baselines["loss"].predict(loss.detach())) * log_prob
            ).mean()

            functional_loss = policy_length_loss + policy_loss - weighted_entropy
            # # if the receiver is deterministic/differentiable, we apply the actual loss
            # TODO in our case the receiver is not trained
            # optimized_loss += loss.mean()

            aux_info["loss_functional"] = functional_loss.clone().reshape(1).detach()
            # aux_info["weighted_entropy"] = weighted_entropy
            # aux_info["policy_loss"] = policy_loss
            # aux_info["policy_length_loss"] = policy_length_loss

            if self.training:
                self.baselines["loss"].update(loss)
                self.baselines["length"].update(length_loss)

            aux_info["sender_entropy"] = entropy_s.detach()
            aux_info["receiver_entropy"] = entropy_r.detach()
            aux_info["length"] = message_lengths.float()  # will be averaged

            optimized_loss += self.weight_functional_loss * functional_loss

            messages = messages.detach()
            receiver_output = receiver_output.detach()

        if self.weight_structural_loss > 0:
            # Forward pass _with_ teacher forcing for structural loss
            _, _, _, scores_struct = sender(
                images_target,
                captions,
                sequence_lengths,
                use_teacher_forcing=True,
                decode_sampling=False,
            )

            loss_struct, _ = loss_structural(captions, scores_struct)
            aux_info["loss_structural"] = loss_struct.reshape(1).detach()

            optimized_loss += self.weight_structural_loss * loss_struct

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            message=messages,
            receiver_output=receiver_output,
            message_length=message_lengths,
            aux=aux_info,
        )

        return optimized_loss, interaction


class OracleSenderReceiverRnnSupervised(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        length_cost: float = 0.0,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks

        """
        super(OracleSenderReceiverRnnSupervised, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost

        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None):
        all_logits, _, message = self.sender(sender_input)
        message_length = find_lengths(message)
        receiver_output, _, _ = self.receiver(message, receiver_input, message_length)

        loss, aux_info = self.loss(
            sender_input, message, all_logits, receiver_input, receiver_output, labels
        )

        optimized_loss = loss.mean()

        aux_info["length"] = message_length.float()  # will be averaged

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )

        return optimized_loss, interaction
