import torch
from torch import nn
import torchvision

from models.image_captioning.captioning_model import CaptioningModel
from models.image_captioning.show_and_tell import ImageEncoder
from models.image_sentence_ranking.ranking_model import l2_norm, ContrastiveLoss
from preprocess import TOKEN_START, TOKEN_END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointLearner(CaptioningModel):
    def __init__(
        self,
        word_embedding_size,
        lstm_hidden_size,
        vocab,
        max_caption_length,
        joint_embeddings_size,
        dropout=0.2,
        fine_tune_resnet=False,
        pretrained_embeddings=None,
        fine_tune_decoder_word_embeddings=True,
    ):
        super(JointLearner, self).__init__(
            vocab,
            word_embedding_size,
            max_caption_length,
            pretrained_embeddings,
            fine_tune_decoder_word_embeddings,
        )
        self.image_encoder = ImageEncoder(joint_embeddings_size, fine_tune_resnet)

        if word_embedding_size != joint_embeddings_size:
            raise ValueError("word embeddings must have same size as joint embeddings")

        self.lstm_hidden_size = lstm_hidden_size

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.word_embedding = nn.Embedding(self.vocab_size, word_embedding_size)

        # Linear layers to find initial states of LSTMs
        self.init_h = nn.Linear(joint_embeddings_size, lstm_hidden_size)
        self.init_c = nn.Linear(joint_embeddings_size, lstm_hidden_size)

        self.lstm = nn.LSTMCell(
            input_size=joint_embeddings_size + word_embedding_size,
            hidden_size=lstm_hidden_size,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden_size, self.vocab_size)

        # Linear transformation for ranking objective
        self.caption_embedding = nn.Linear(lstm_hidden_size, joint_embeddings_size)

        self.loss_ranking = ContrastiveLoss()

    def init_hidden_states(self, encoder_output):
        """
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_output.squeeze(1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        states = [h, c]
        return states

    def lstm_input_first_timestep(self, batch_size, encoder_output):
        # At the start, all 'previous words' are the <start> token
        start_tokens = torch.full(
            (batch_size,), self.vocab[TOKEN_START], dtype=torch.int64, device=device
        )
        return self.word_embedding(start_tokens)

    def forward(self, images, target_captions=None, decode_lengths=None):
        """
        Forward propagation.

        :param images: input images
        :param target_captions: encoded target captions, shape: (batch_size, max_caption_length)
        :param decode_lengths: caption lengths, shape: (batch_size, 1)
        :return: scores for vocabulary, decode lengths, weights
        """
        encoder_output = self.image_encoder(images)

        batch_size = encoder_output.size(0)

        # Flatten image
        images_embedded = encoder_output.view(batch_size, -1, encoder_output.size(-1))

        use_teacher_forcing = False
        if decode_lengths is not None:
            use_teacher_forcing = True

            # Do not decode at last timestep (after EOS token)
            decode_lengths = decode_lengths - 1
        else:
            decode_lengths = torch.full(
                (batch_size,),
                self.max_caption_length,
                dtype=torch.int64,
                device=device,
            )

        # Initialize LSTM state
        states = self.init_hidden_states(images_embedded)

        # Tensors to hold word prediction scores
        scores = torch.zeros(
            (batch_size, max(decode_lengths), self.vocab_size), device=device
        )

        # Tensor to store language encodings for sentence embeddings
        lang_enc_hidden_activations = torch.zeros(
            (batch_size, self.lstm_hidden_size), device=device
        )

        for t in range(max(decode_lengths)):
            if not use_teacher_forcing and not t == 0:
                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = (
                    torch.nonzero(prev_words == self.vocab[TOKEN_END])
                    .view(-1)
                    .tolist()
                )

                # Update the decode lengths accordingly
                decode_lengths[ind_end_token] = torch.min(
                    decode_lengths[ind_end_token],
                    torch.full_like(decode_lengths[ind_end_token], t, device=device),
                )

            # Check if all sequences are finished:
            indices_incomplete_sequences = torch.nonzero(decode_lengths > t).view(-1)
            if len(indices_incomplete_sequences) == 0:
                break

            if t == 0:
                prev_words_embedded = self.lstm_input_first_timestep(batch_size, images_embedded)
            else:
                prev_words_embedded = self.word_embedding(prev_words)

            scores_for_timestep, states, _ = self.forward_step(
                images_embedded, prev_words_embedded, states
            )

            # Update the previously predicted words
            prev_words = self.update_previous_word(
                scores_for_timestep, target_captions, t, use_teacher_forcing
            )

            # Store scores
            scores[indices_incomplete_sequences, t, :] = scores_for_timestep[
                indices_incomplete_sequences
            ]

            # Store last hidden activations of LSTM for finished sequences
            h_lan_enc = states[0]
            lang_enc_hidden_activations[decode_lengths == t + 1] = h_lan_enc[
                decode_lengths == t + 1
            ]

        captions_embedded = self.caption_embedding(lang_enc_hidden_activations)
        captions_embedded = l2_norm(captions_embedded)

        images_embedded = l2_norm(images_embedded.squeeze(1))

        return scores, decode_lengths, None, images_embedded, captions_embedded

    def forward_step(self, encoder_output, prev_word_embeddings, states):
        """Perform a single decoding step."""
        decoder_hidden_state, decoder_cell_state = states

        encoder_output = encoder_output.squeeze(1)

        lstm_input = torch.cat((encoder_output, prev_word_embeddings), dim=1)
        decoder_hidden_state, decoder_cell_state = self.lstm(
            lstm_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(decoder_hidden_state)

        states = [decoder_hidden_state, decoder_cell_state]
        return scores, states, None

    def loss(self, scores, target_captions, decode_lengths, alphas, images_embedded, captions_embedded, reduction="mean"):
        # Image captioning loss
        loss_captioning = self.loss_cross_entropy(scores, target_captions, decode_lengths, reduction)

        # Ranking loss
        loss_ranking = self.loss_ranking(images_embedded, captions_embedded)

        return loss_captioning, loss_ranking


