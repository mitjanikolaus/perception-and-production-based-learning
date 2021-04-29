import os
import pickle

import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torchtext.vocab import Vocab
from torchvision.models import resnet50
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from egg.core import RnnSenderReinforce
from preprocess import TOKEN_START, TOKEN_END, TOKEN_PADDING
from utils import SPECIAL_CHARACTERS, sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VisualRefSpeakerDiscriminativeOracle(nn.Module):
    def __init__(self, data_folder, captions_filename, max_sequence_length, vocab):
        super(VisualRefSpeakerDiscriminativeOracle, self).__init__()

        # Load captions
        with open(os.path.join(data_folder, captions_filename["train"]), "rb") as file:
            self.captions_train = pickle.load(file)

        with open(os.path.join(data_folder, captions_filename["val"]), "rb") as file:
            self.captions_val = pickle.load(file)

        with open(os.path.join(data_folder, captions_filename["test"]), "rb") as file:
            self.captions_test = pickle.load(file)

        self.max_sequence_length = max_sequence_length

        self.vocab = vocab
        self.ignore_tokens = [vocab.stoi[s] for s in [Vocab.UNK] + SPECIAL_CHARACTERS]

    def pad_messages(self, messages, padding_value=0.0):
        """Trim and pad all messages to max sequence length."""
        trailing_dims = messages[0].size()[1:]
        max_len = self.max_sequence_length
        out_dims = (len(messages), max_len) + trailing_dims

        out_tensor = messages[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(messages):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :length, ...] = tensor[: self.max_sequence_length]

        return out_tensor

    def forward(self, input):
        images, target_label, target_image_ids, distractor_image_ids = input

        # pick the target's caption that has the least word overlap with any of the distractor's captions
        # (the score is normalized by the captions’ length excluding stop-words)
        def get_relevant_tokens(caption):
            return set([t for t in caption if t not in self.ignore_tokens])

        if self.training:
            captions = self.captions_train
        else:
            # TODO test mode
            captions = self.captions_val

        output_captions = []
        for target_image_id, distractor_image_id in zip(
            target_image_ids, distractor_image_ids
        ):
            overlap_scores = []
            for target_caption in captions[int(target_image_id)]:
                target_caption_tokens = get_relevant_tokens(target_caption)

                distractor_captions = [
                    t for caption in captions[int(distractor_image_id)] for t in caption
                ]
                distractor_captions_tokens = get_relevant_tokens(distractor_captions)

                overlap = len(target_caption_tokens & distractor_captions_tokens)
                overlap_scores.append(overlap / len(target_caption_tokens))

            best_caption_idx = np.argmin(overlap_scores)
            best_caption = captions[int(target_image_id)][best_caption_idx]
            output_captions.append(best_caption)

        # append end of message token
        for caption in output_captions:
            caption.append(0)

        # Transform lists to tensors
        output_captions = [
            torch.tensor(caption, device=device) for caption in output_captions
        ]

        # Pad all captions in batch to equal length
        output_captions = self.pad_messages(output_captions)

        logits = torch.zeros_like(output_captions).to(device)
        entropy = logits

        return output_captions, logits, entropy


class VisualRefListenerOracle(nn.Module):
    def __init__(self, ranking_model):
        super(VisualRefListenerOracle, self).__init__()
        self.ranking_model = ranking_model

    def forward(self, messages, receiver_input, message_lengths):
        batch_size = receiver_input[0].shape[0]

        images_1, images_2 = receiver_input[0], receiver_input[1]

        images_1_embedded, messages_embedded = self.ranking_model(
            images_1, messages, message_lengths
        )
        images_2_embedded, messages_embedded = self.ranking_model(
            images_2, messages, message_lengths
        )

        stacked_images = torch.stack([images_1_embedded, images_2_embedded], dim=1)

        similarities = torch.matmul(
            stacked_images, torch.unsqueeze(messages_embedded, dim=-1)
        )

        logits = torch.zeros(batch_size).to(device)
        entropy = logits

        # out: scores, logits, entropy
        return similarities.view(batch_size, -1), logits, entropy


class ImageEncoder(nn.Module):
    def __init__(self, visual_embedding_size, fine_tune_resnet=False):
        super(ImageEncoder, self).__init__()

        resnet = resnet50(pretrained=True)

        if not fine_tune_resnet:
            for param in resnet.parameters():
                param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, visual_embedding_size)

    def forward(self, input):
        image_features = self.resnet(input)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.embed(image_features)

        return image_features


CELL_TYPES = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}


class RnnSenderMultitaskVisualRef(RnnSenderReinforce):
    def __init__(
        self,
        agent,
        vocab,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
        cell="rnn",
    ):
        super(RnnSenderMultitaskVisualRef, self).__init__(
            agent, vocab_size, embed_dim, hidden_size, max_len, num_layers, cell
        )

        self.vocab = vocab

        cell_type = CELL_TYPES[cell]
        # Expand input dimension of RNN/LSTM cells so that we can feed image input at every timestep
        self.cells = nn.ModuleList(
            [
                cell_type(input_size=embed_dim + hidden_size, hidden_size=hidden_size)
                if i == 0
                else cell_type(
                    input_size=hidden_size + hidden_size, hidden_size=hidden_size
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(
        self,
        images_target,
        captions=None,
        decode_lengths=None,
        use_teacher_forcing=True,
        decode_sampling=False,
    ):
        if captions is None:
            use_teacher_forcing = False
            decode_sampling = True

        batch_size = images_target.size(0)

        if use_teacher_forcing:
            # Do not decode at last timestep (after EOS token)
            decode_lengths = decode_lengths - 1
        else:
            decode_lengths = torch.full(
                (batch_size,), self.max_len - 1, dtype=torch.int64, device=device,
            )

        image_features = self.agent(images_target)
        prev_hidden = [image_features]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        prev_c = [
            torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * prev_hidden[0].size(0))

        # TODO: do the same for logits and entropy, reduce computational overhead!
        sequences = torch.zeros(
            (batch_size, max(decode_lengths)), device=device, dtype=torch.long
        )

        logits = []
        entropy = []

        for step in range(max(decode_lengths)):
            if not use_teacher_forcing and not step == 0:
                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = (
                    torch.nonzero(x == self.vocab[TOKEN_END]).view(-1).tolist()
                )

                # Update the decode lengths accordingly
                decode_lengths[ind_end_token] = torch.min(
                    decode_lengths[ind_end_token],
                    torch.full_like(decode_lengths[ind_end_token], step, device=device),
                )
            # # Check if all sequences are finished:
            indices_incomplete_sequences = torch.nonzero(decode_lengths > step).view(-1)
            # TODO do not break because we want all messages to be same length for logging..
            # if len(indices_incomplete_sequences) == 0:
            #     break

            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    # TODO: give image input at every LSTM layer?
                    lstm_input = torch.cat((image_features, input), dim=1)
                    h_t, c_t = layer(lstm_input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if decode_sampling:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            sequences[indices_incomplete_sequences, step] = x[
                indices_incomplete_sequences
            ]

            if use_teacher_forcing:
                x_gold = captions[:, step + 1]
                input = self.embedding(x_gold)
            else:
                input = self.embedding(x)

        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequences.size(0), 1)).to(sequences.device)

        sequences = torch.cat([sequences, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        # Add SOS token to messages
        sos_tokens = torch.full((batch_size, 1), self.vocab.stoi[TOKEN_START], dtype=torch.long, device=device)
        sequences = torch.cat([sos_tokens, sequences], dim=1)

        return sequences, logits, entropy

    def decode(self, x, num_samples):
        batch_size = x.shape[0]
        if batch_size != 1:
            raise RuntimeError("Decoding with batch size greater 1 not implemented.")

        # Repeat input to obtain multiple samples
        x = x.repeat(num_samples, 1, 1, 1)
        scores, sequence_lengths, extra = self.forward(
            x, use_teacher_forcing=False, decode_sampling=True
        )

        return sequences(scores), sequence_lengths, extra

    def perplexity(self, images, captions, caption_lengths):
        """Return perplexities of captions given images."""

        scores, _, _ = self.forward(images, captions, caption_lengths)

        # Do not reduce among samples in batch
        loss = loss_cross_entropy(scores, captions, reduction="none")

        perplexities = torch.exp(loss)

        # sum up cross entropies of all words
        perplexities = perplexities.sum(dim=1)

        return perplexities


def loss_cross_entropy(scores, target_captions, reduction="mean", ignore_index=0):
    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    target_captions = target_captions[:, 1:]

    # Trim produced captions to max target length
    scores = scores[:, : target_captions.shape[1]]

    if scores.shape[1] < target_captions.shape[1]:
        # Model didn't produce sequences as long as gold, so we cut the gold captions to be able to calculate the loss
        target_captions = target_captions[:, : scores.shape[1]]

    scores = scores.permute(0, 2, 1)
    return F.cross_entropy(
        scores, target_captions, ignore_index=ignore_index, reduction=reduction,
    )
