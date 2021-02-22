from typing import Optional, Dict

import torch
from dataclasses import dataclass

from egg.core import Interaction, LoggingStrategy
from preprocess import TOKEN_START, TOKEN_END, TOKEN_PADDING

import matplotlib.pyplot as plt

SPECIAL_CHARACTERS = [TOKEN_START, TOKEN_END, TOKEN_PADDING]

def decode_caption(caption, vocab):
    words = [vocab.itos[word] for word in caption if vocab.itos[word] not in SPECIAL_CHARACTERS]
    return " ".join(words)


def print_caption(caption, vocab):
    caption = decode_caption(caption, vocab)
    print(caption)

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
        target_image, distractor_image, target_image_id, distractor_image_id = sender_input
        filtered_sender_input = torch.stack((target_image_id, distractor_image_id))

        return Interaction(
            sender_input=filtered_sender_input,
            receiver_input=None,
            labels=labels,
            message=message,
            receiver_output=receiver_output,
            message_length=message_length,
            aux=aux,
        )
