import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F

from models.image_captioning.captioning_model import CaptioningModel
from preprocess import TOKEN_PADDING, TOKEN_START

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    def __init__(self, visual_embedding_size, fine_tune_resnet=False):
        super(ImageEncoder, self).__init__()

        resnet = resnet50(pretrained=True)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, visual_embedding_size)

        self.set_fine_tuning_enabled(fine_tune_resnet)

    def forward(self, images):
        """
        Forward propagation.

        :return: encoded images
        """
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.embed(image_features)

        return image_features

    def set_fine_tuning_enabled(self, enable_fine_tuning):
        """

        :param enable_fine_tuning: Set to True to enable fine tuning
        """
        for param in self.resnet.parameters():
            param.requires_grad = enable_fine_tuning


class ShowAndTell(CaptioningModel):
    def __init__(
        self,
        word_embedding_size,
        visual_embeddings_size,
        lstm_hidden_size,
        vocab,
        max_caption_length,
        dropout=0.2,
        fine_tune_resnet=False,
        pretrained_embeddings=None,
        fine_tune_decoder_word_embeddings=True,
    ):
        super(ShowAndTell, self).__init__(
            vocab,
            word_embedding_size,
            max_caption_length,
            pretrained_embeddings,
            fine_tune_decoder_word_embeddings,
        )
        self.image_encoder = ImageEncoder(visual_embeddings_size, fine_tune_resnet)

        self.lstm_hidden_size = lstm_hidden_size

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.word_embedding = nn.Embedding(self.vocab_size, word_embedding_size)

        # Linear layers to find initial states of LSTMs
        self.init_h = nn.Linear(visual_embeddings_size, lstm_hidden_size)
        self.init_c = nn.Linear(visual_embeddings_size, lstm_hidden_size)

        self.lstm = nn.LSTMCell(
            input_size=visual_embeddings_size + word_embedding_size,
            hidden_size=lstm_hidden_size,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden_size, self.vocab_size)

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

    def loss(self, scores, target_captions, decode_lengths, alphas, reduction="mean"):
        loss = self.loss_cross_entropy(
            scores, target_captions, decode_lengths, reduction
        )

        return loss
