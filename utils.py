import os
from pathlib import Path
from typing import Optional, Dict

import torch
from dataclasses import dataclass

from egg.core import Interaction, LoggingStrategy
from preprocess import TOKEN_START, TOKEN_END, TOKEN_PADDING

SPECIAL_CHARACTERS = [TOKEN_START, TOKEN_END, TOKEN_PADDING]

CHECKPOINT_DIR_IMAGE_CAPTIONING = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/captioning/"
)

CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/ranking/image_sentence_ranking.pt"
)
SEMANTIC_ACCURACIES_PATH_RANKING = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/ranking/semantic_accuracies_ranking.p"
)

CHECKPOINT_PATH_LANGUAGE_MODEL_BEST = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/lm/language_model.pt"
)
SEMANTIC_ACCURACIES_PATH_LANGUAGE_MODEL = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/lm/semantic_accuracies_lm.p"
)

SEMANTICS_EVAL_FILES = [
    "data/semantics_eval_persons.csv",
    "data/semantics_eval_animals.csv",
    "data/semantics_eval_inanimates.csv",
    "data/semantics_eval_verbs.csv",
    "data/semantics_eval_adjectives.csv",
    "data/semantics_eval_adjective_noun_binding.csv",
    "data/semantics_eval_agent_patient_filtered.csv",
]


def decode_caption(caption, vocab, join=True):
    words = [
        vocab.itos[word]
        for word in caption
        if vocab.itos[word] not in SPECIAL_CHARACTERS
    ]
    if join:
        return " ".join(words)
    else:
        return words


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
        (
            target_image,
            distractor_image,
            target_image_id,
            distractor_image_id,
        ) = sender_input
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
