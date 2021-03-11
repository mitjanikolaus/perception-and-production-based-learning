import torch
from torch import nn
import torchvision

from models.image_captioning.captioning_model import CaptioningModel
from models.image_sentence_ranking.ranking_model import l2_norm, ContrastiveLoss
from preprocess import TOKEN_START, TOKEN_END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    def __init__(self, fine_tune_resnet=False, encoded_image_size=14):
        super(ImageEncoder, self).__init__()

        resnet = torchvision.models.resnet152(pretrained=True)

        # Remove linear and pool layers, these are only used for classification
        modules = list(resnet.children())[:-2]
        self.model = nn.Sequential(*modules)

        # Resize input image to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        # Disable calculation of all gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        if fine_tune_resnet:
            self.set_fine_tuning_enabled(fine_tune_resnet)

    def forward(self, images):
        """
        Forward propagation.

        :param images: input images, shape: (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(
            images
        )  # output shape: (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(
            out
        )  # output shape: (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(
            0, 2, 3, 1
        )  # output shape: (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def set_fine_tuning_enabled(self, enable_fine_tuning):
        """
        Enable or disable the computation of gradients for the convolutional blocks 2-4 of the encoder.

        :param enable_fine_tuning: Set to True to enable fine tuning
        """
        # The convolutional blocks 2-4 are found at position 5-7 in the model
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = enable_fine_tuning


class JointLearnerSAT(CaptioningModel):
    """Joint learner with attention (based on the Show, Attend and Tell model)."""

    ENCODER_DIM = 2048
    ATTENTION_DIM = 512
    ALPHA_C = 1.0

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
        super(JointLearnerSAT, self).__init__(
            vocab,
            word_embedding_size,
            max_caption_length,
            pretrained_embeddings,
            fine_tune_decoder_word_embeddings,
        )
        self.image_encoder = ImageEncoder(fine_tune_resnet)

        self.image_embedding = nn.Linear(self.ENCODER_DIM, joint_embeddings_size)

        self.attention = AttentionModule(
            lstm_hidden_size, self.ATTENTION_DIM,
        )

        # Linear layers to find initial states of LSTMs
        self.init_h = nn.Linear(joint_embeddings_size, lstm_hidden_size)
        self.init_c = nn.Linear(joint_embeddings_size, lstm_hidden_size)

        # Gating scalars and sigmoid layer (cf. section 4.2.1 of the paper)
        self.f_beta = nn.Linear(lstm_hidden_size, joint_embeddings_size)
        self.sigmoid = nn.Sigmoid()

        # LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTMCell(
            word_embedding_size + joint_embeddings_size, lstm_hidden_size,
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Linear layers for output generation
        self.linear_o = nn.Linear(word_embedding_size, self.vocab_size)
        self.linear_h = nn.Linear(lstm_hidden_size, word_embedding_size)
        self.linear_z = nn.Linear(joint_embeddings_size, word_embedding_size)

        # Linear transformation for ranking objective
        self.caption_embedding = nn.Linear(lstm_hidden_size, joint_embeddings_size)

        self.loss_ranking = ContrastiveLoss()

    def init_hidden_states(self, encoder_out):
        """
        Create the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
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
        encoder_output = encoder_output.view(batch_size, -1, encoder_output.size(-1))

        # Embed image
        images_embedded = self.image_embedding(encoder_output)

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

        # Tensors to hold word prediction scores and alphas
        scores = torch.zeros(
            (batch_size, max(decode_lengths), self.vocab_size), device=device
        )
        alphas = torch.zeros(
            batch_size, max(decode_lengths), images_embedded.size(1), device=device
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

            scores_for_timestep, states, alphas_for_timestep = self.forward_step(
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

            if alphas_for_timestep is not None:
                alphas[indices_incomplete_sequences, t, :] = alphas_for_timestep[
                    indices_incomplete_sequences
                ]

        captions_embedded = self.caption_embedding(lang_enc_hidden_activations)
        captions_embedded = l2_norm(captions_embedded)

        images_embedded_mean = images_embedded.mean(dim=1)

        return scores, decode_lengths, alphas, images_embedded_mean, captions_embedded


    def forward_step(self, encoder_output, prev_word_embeddings, states):
        """Perform a single decoding step."""
        decoder_hidden_state, decoder_cell_state = states

        attention_weighted_encoding, alpha = self.attention(
            encoder_output, decoder_hidden_state
        )
        gating_scalars = self.sigmoid(self.f_beta(decoder_hidden_state))
        attention_weighted_encoding = gating_scalars * attention_weighted_encoding

        decoder_input = torch.cat(
            (prev_word_embeddings, attention_weighted_encoding), dim=1
        )
        decoder_hidden_state, decoder_cell_state = self.lstm(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        decoder_hidden_state_embedded = self.linear_h(decoder_hidden_state)
        attention_weighted_encoding_embedded = self.linear_z(
            attention_weighted_encoding
        )
        scores = self.linear_o(
            self.dropout(
                prev_word_embeddings
                + decoder_hidden_state_embedded
                + attention_weighted_encoding_embedded
            )
        )

        states = [decoder_hidden_state, decoder_cell_state]
        return scores, states, alpha

    def loss(self, scores, target_captions, decode_lengths, alphas, images_embedded, captions_embedded, reduction="mean"):
        # Image captioning loss
        loss_captioning = self.loss_cross_entropy(scores, target_captions, decode_lengths, reduction)

        # Add doubly stochastic attention regularization
        loss_captioning += self.ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        # Ranking loss
        loss_ranking = self.loss_ranking(images_embedded, captions_embedded)

        return loss_captioning, loss_ranking


class AttentionModule(nn.Module):
    def __init__(self, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param attention_dim: size of the attention network
        """
        super(AttentionModule, self).__init__()

        # Linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        # Linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)

        # ReLU layer
        self.relu = nn.ReLU()

        # Softmax layer to calculate attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, shape: (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att2 = self.decoder_att(
            decoder_hidden
        )  # output shape: (batch_size, attention_dim)
        att = self.full_att(self.relu(encoder_out + att2.unsqueeze(1))).squeeze(
            2
        )  # output shape: (batch_size, num_pixels)
        alpha = self.softmax(att)  # output shape: (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # output shape: (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
