import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet50

import numpy as np

from models.image_sentence_ranking.ranking_model import l2_norm, ImageEmbedding, cosine_sim, ContrastiveLoss, \
    ImageSentenceRanker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageSentenceRankerGrounded(ImageSentenceRanker):
    def __init__(
        self,
        word_embedding_size,
        joint_embeddings_size,
        lstm_hidden_size,
        vocab_size,
        fine_tune_resnet=False,
    ):
        super(ImageSentenceRankerGrounded, self).__init__(word_embedding_size, joint_embeddings_size, lstm_hidden_size, vocab_size, fine_tune_resnet)

        self.language_encoding_lstm = LanguageEncodingLSTM(
            word_embedding_size, joint_embeddings_size, lstm_hidden_size,
        )

    def forward(self, encoder_output, captions, caption_lengths):
        """
        Forward propagation for the ranking task.
        """
        images_embedded = self.image_embedding(encoder_output)
        captions_embedded = self.embed_captions_grounded(captions, caption_lengths, images_embedded)

        return images_embedded, captions_embedded

    def embed_captions_grounded(self, captions, decode_lengths, images_embedded):
        # Initialize LSTM state
        batch_size = captions.size(0)
        h_lan_enc, c_lan_enc = self.language_encoding_lstm.init_state(batch_size)

        # Tensor to store hidden activations
        lang_enc_hidden_activations = torch.zeros(
            (batch_size, self.lstm_hidden_size), device=device
        )

        for t in range(max(decode_lengths)):
            prev_words_embedded = self.word_embedding(captions[:, t])

            h_lan_enc, c_lan_enc = self.language_encoding_lstm(
                h_lan_enc, c_lan_enc, prev_words_embedded, images_embedded
            )

            lang_enc_hidden_activations[decode_lengths == t + 1] = h_lan_enc[
                decode_lengths == t + 1
            ]

        captions_embedded = self.caption_embedding(lang_enc_hidden_activations)
        captions_embedded = l2_norm(captions_embedded)
        return captions_embedded


class LanguageEncodingLSTM(nn.Module):
    def __init__(self, word_embeddings_size, joint_embeddings_size, hidden_size):
        super(LanguageEncodingLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(word_embeddings_size+joint_embeddings_size, hidden_size)

    def forward(self, h, c, prev_words_embedded, images_embedded):
        lstm_input = torch.cat((images_embedded, prev_words_embedded), dim=1)

        h_out, c_out = self.lstm_cell(lstm_input, (h, c))
        return h_out, c_out

    def init_state(self, batch_size):
        h = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        c = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        return [h, c]

