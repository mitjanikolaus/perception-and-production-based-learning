import os
from pathlib import Path
import random

import numpy as np

import torch
from torch.distributions import Categorical

from preprocess import TOKEN_START, TOKEN_END, TOKEN_PADDING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_LOG_FREQUENCY = 100
DEFAULT_BATCH_SIZE = 100

DEFAULT_MAX_NUM_VAL_SAMPLES = 1000

DEFAULT_WORD_EMBEDDINGS_SIZE = 100
DEFAULT_LSTM_HIDDEN_SIZE: int = 512

SPECIAL_CHARACTERS = [TOKEN_START, TOKEN_END, TOKEN_PADDING]

CHECKPOINT_DIR_IMAGE_CAPTIONING = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/captioning/"
)

CHECKPOINT_DIR_RANKING = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/ranking/"
)

CHECKPOINT_PATH_LANGUAGE_MODEL_BEST = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/lm/language_model.pt"
)
SEMANTIC_ACCURACIES_PATH_LANGUAGE_MODEL = os.path.join(
    Path.home(), "data/visual_ref/checkpoints/lm/semantic_accuracies_lm.p"
)

CHECKPOINT_NAME_SENDER = "sender.pt"
CHECKPOINT_NAME_RECEIVER = "receiver.pt"

SEMANTICS_EVAL_FILES = [
    "data/semantics_eval_persons.csv",
    "data/semantics_eval_animals.csv",
    "data/semantics_eval_inanimates.csv",
    "data/semantics_eval_verbs.csv",
    "data/semantics_eval_adjectives.csv",
    "data/semantics_eval_adjective_noun_binding.csv",
    "data/semantics_eval_verb_noun_binding_filtered.csv",
    "data/semantics_eval_semantic_roles_filtered.csv",
]

LEGEND = {
    "data/semantics_eval_persons.csv": "persons",
    "data/semantics_eval_animals.csv": "animals",
    "data/semantics_eval_inanimates.csv": "objects",
    "data/semantics_eval_verbs.csv": "verbs",
    "data/semantics_eval_adjectives.csv": "adjectives",
    "data/semantics_eval_adjective_noun_binding.csv": "adjective-noun dependency",
    "data/semantics_eval_verb_noun_binding_filtered.csv": "verb-noun dependency",
    "data/semantics_eval_semantic_roles_filtered.csv": "semantic roles",
}

LEGEND_GROUPED_NOUNS = {
    "nouns": "nouns",
    "data/semantics_eval_verbs.csv": "verbs",
    "data/semantics_eval_adjectives.csv": "adjectives",
    "data/semantics_eval_adjective_noun_binding.csv": "adjective-noun dependency",
    "data/semantics_eval_verb_noun_binding_filtered.csv": "verb-noun dependency",
    "data/semantics_eval_semantic_roles_filtered.csv": "semantic roles",
}

def sequences(scores, pad_to_length=None):
    """Get the most likely sequence given scores."""
    seq = scores.argmax(dim=2)
    if pad_to_length:
        seq = torch.stack(
            [torch.cat([s, torch.zeros(pad_to_length - len(s), dtype=torch.long, device=device)]) for s in seq]
        )
    return seq


def entropy(scores):
    """Get the entropies for each sample and each timestep from scores."""
    return torch.stack(
        [
            Categorical(logits=x_i).entropy()
            for i, x_i in enumerate(torch.unbind(scores, dim=1), 0)
        ],
        dim=1,
    ).T


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


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
