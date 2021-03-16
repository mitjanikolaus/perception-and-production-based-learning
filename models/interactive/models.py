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
from preprocess import TOKEN_START, TOKEN_END
from utils import SPECIAL_CHARACTERS

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
            out_tensor[i, :length, ...] = tensor[:self.max_sequence_length]

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
        for target_image_id, distractor_image_id in zip(target_image_ids, distractor_image_ids):
            overlap_scores = []
            for target_caption in captions[int(target_image_id)]:
                target_caption_tokens = get_relevant_tokens(target_caption)

                distractor_captions = [t for caption in captions[int(distractor_image_id)] for t in caption]
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
        output_captions = [torch.tensor(caption, device=device) for caption in output_captions]

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

        images_1_embedded, messages_embedded = self.ranking_model(images_1, messages, message_lengths)
        images_2_embedded, messages_embedded = self.ranking_model(images_2, messages, message_lengths)

        stacked_images = torch.stack([images_1_embedded, images_2_embedded], dim=1)

        similarities = torch.matmul(stacked_images, torch.unsqueeze(messages_embedded, dim=-1))

        logits = torch.zeros(batch_size).to(device)
        entropy = logits

        # out: scores, logits, entropy
        return similarities.view(batch_size, -1),  logits, entropy


class VisualRefSenderFunctional(nn.Module):
    def __init__(self, visual_embedding_size, fine_tune_resnet=False):
        super(VisualRefSenderFunctional, self).__init__()

        resnet = resnet50(pretrained=True)

        if not fine_tune_resnet:
            for param in resnet.parameters():
                param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, visual_embedding_size)

    def forward(self, input):
        images, target_label, target_image_ids, distractor_image_ids, captions, sequence_lengths = input

        # TODO: verify
        images_target = images[target_label, range(images.size(1))]

        image_features = self.resnet(images_target)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.embed(image_features)

        # output is used to initialize the message producing RNN
        return image_features, captions, sequence_lengths


class RnnSenderMultitaskVisualRef(RnnSenderReinforce):

    def forward(self, x):
        image_features, captions, sequence_lengths = self.agent(x)
        prev_hidden = [image_features]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        prev_c = [
            torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * prev_hidden[0].size(0))

        sequence = []
        logits = []
        entropy = []
        all_logits = []

        max_len = captions.size(1) if self.training else self.max_len
        for step in range(max_len - 1):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            all_logits.append(step_logits)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                # x = distr.sample()
                # Use teacher forcing during training
                x = captions[:, step + 1]
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy, all_logits


