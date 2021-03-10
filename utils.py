import os
from pathlib import Path

from preprocess import TOKEN_START, TOKEN_END, TOKEN_PADDING

DEFAULT_LOG_FREQUENCY = 100
DEFAULT_BATCH_SIZE = 32

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

