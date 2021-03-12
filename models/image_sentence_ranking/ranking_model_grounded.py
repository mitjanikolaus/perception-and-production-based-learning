import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet50

import numpy as np

from models.image_sentence_ranking.ranking_model import l2_norm, ImageEmbedding, cosine_sim, \
    ImageSentenceRanker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastiveLossAlternative(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, max_violation=False):
        super(ContrastiveLossAlternative, self).__init__()

        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images_embedded, captions_embedded):
        # compute image-caption score matrix
        scores = cosine_sim(images_embedded, captions_embedded)
        diagonal = scores.diag().view(images_embedded.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        # cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask).to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            # cost_im = cost_im.max(0)[0]

        # Sum up caption retrieval and image retrieval loss
        sum_of_losses = cost_s.sum() #+ cost_im.sum()

        # Normalize loss by batch size
        normalized_loss = sum_of_losses / images_embedded.size(0)

        return normalized_loss


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

        self.loss = ContrastiveLossAlternative()


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

