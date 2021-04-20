import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet50

import numpy as np

from preprocess import TOKEN_PADDING

import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageModel(nn.Module):
    def __init__(
        self, word_embedding_size, lstm_hidden_size, vocab, dropout=0.2,
    ):
        super(LanguageModel, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.word_embedding = nn.Embedding(self.vocab_size, word_embedding_size)

        self.lstm = nn.LSTM(word_embedding_size, lstm_hidden_size, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(lstm_hidden_size, self.vocab_size)

        self.loss = nn.CrossEntropyLoss()

    def loss(self, scores, target_captions, reduction="mean"):
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        target_captions = target_captions[:, 1:]

        return F.cross_entropy(
            scores,
            target_captions,
            ignore_index=self.vocab[TOKEN_PADDING],
            reduction=reduction,
        )

    def perplexity(self, captions, caption_lengths):
        """Return perplexities of captions given images."""

        scores = self.forward(captions, caption_lengths)

        loss = self.loss(scores, captions, reduction="none")

        # sum up cross entropies of all words
        perplexities = torch.exp(loss)

        perplexities = perplexities.sum(dim=1)

        return perplexities

    def forward(self, captions, caption_lengths):
        """
        Forward propagation.
        """
        # Do not decode at last timestep (after EOS token)
        decode_lengths = caption_lengths - 1

        embedded = self.word_embedding(captions)

        packed_input = pack_padded_sequence(
            embedded, decode_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.lstm(packed_input)

        out, lengths = pad_packed_sequence(packed_output, batch_first=True)

        out = self.dropout(out)

        scores = self.fc(out)

        return scores.permute(0, 2, 1)


class LanguageEncodingLSTM(nn.Module):
    def __init__(self, word_embeddings_size, hidden_size):
        super(LanguageEncodingLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(word_embeddings_size, hidden_size)

    def forward(self, h, c, prev_words_embedded):
        h_out, c_out = self.lstm_cell(prev_words_embedded, (h, c))
        return h_out, c_out
