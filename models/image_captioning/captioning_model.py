import random

import torch
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from preprocess import TOKEN_PADDING
from utils import TOKEN_START, decode_caption, TOKEN_END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptioningModel(nn.Module):

    TEACHER_FORCING_RATIO = 1

    def __init__(
        self,
        vocab,
        word_embedding_size,
        max_caption_length,
        pretrained_embeddings=None,
        fine_tune_decoder_word_embeddings=True,
    ):
        super(CaptioningModel, self).__init__()

        self.vocab_size = len(vocab)
        self.vocab = vocab

        self.max_caption_length = max_caption_length

        self.word_embedding = nn.Embedding(self.vocab_size, word_embedding_size)

        if pretrained_embeddings is not None:
            self.word_embedding.weight = nn.Parameter(pretrained_embeddings)
        self.set_fine_tune_embeddings(fine_tune_decoder_word_embeddings)

    def set_fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of the embedding layer.

        :param fine_tune: Set to True to allow fine tuning
        """
        for p in self.word_embedding.parameters():
            p.requires_grad = fine_tune

    def update_previous_word(self, scores, target_words, t, use_teacher_forcing):
        if use_teacher_forcing:
            if random.random() < self.TEACHER_FORCING_RATIO:
                use_teacher_forcing = True
            else:
                use_teacher_forcing = False

        if use_teacher_forcing:
            next_words = target_words[:, t + 1]
        else:
            next_words = torch.argmax(scores, dim=1)

        return next_words

    def lstm_input_first_timestep(self, batch_size, encoder_output):
        raise NotImplementedError()

    def forward(self, images, target_captions=None, decode_lengths=None):
        """
        Forward propagation.

        :param images: input images
        :param target_captions: encoded target captions, shape: (batch_size, max_caption_length)
        :param decode_lengths: caption lengths, shape: (batch_size, 1)
        :return: scores for vocabulary, decode lengths, weights
        """
        use_teacher_forcing = False
        if decode_lengths is not None:
            use_teacher_forcing = True

            # Do not decode at last timestep (after EOS token)
            decode_lengths = decode_lengths - 1

        encoder_output = self.image_encoder(images)

        batch_size = encoder_output.size(0)

        # Flatten image
        encoder_output = encoder_output.view(batch_size, -1, encoder_output.size(-1))

        if not use_teacher_forcing:
            decode_lengths = torch.full(
                (batch_size,),
                self.max_caption_length,
                dtype=torch.int64,
                device=device,
            )

        # Initialize LSTM state
        states = self.init_hidden_states(encoder_output)

        # Tensors to hold word prediction scores and alphas
        scores = torch.zeros(
            (batch_size, max(decode_lengths), self.vocab_size), device=device
        )
        alphas = torch.zeros(
            batch_size, max(decode_lengths), encoder_output.size(1), device=device
        )

        for t in range(max(decode_lengths)):
            if not use_teacher_forcing and not t == 0:
                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = (
                    torch.nonzero(prev_words == self.vocab[TOKEN_END]).view(-1).tolist()
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
                prev_words_embedded = self.lstm_input_first_timestep(
                    batch_size, encoder_output
                )
            else:
                prev_words_embedded = self.word_embedding(prev_words)

            scores_for_timestep, states, alphas_for_timestep = self.forward_step(
                encoder_output, prev_words_embedded, states
            )

            # Update the previously predicted words
            prev_words = self.update_previous_word(
                scores_for_timestep, target_captions, t, use_teacher_forcing
            )

            scores[indices_incomplete_sequences, t, :] = scores_for_timestep[
                indices_incomplete_sequences
            ]
            if alphas_for_timestep is not None:
                alphas[indices_incomplete_sequences, t, :] = alphas_for_timestep[
                    indices_incomplete_sequences
                ]

        return scores, decode_lengths, alphas

    def loss_cross_entropy(self, scores, target_captions, decode_lengths, reduction):
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        target_captions = target_captions[:, 1:]

        # Trim produced captions to max target length
        scores = scores[:, : target_captions.shape[1]]

        scores = scores.permute(0, 2, 1)
        return F.cross_entropy(
            scores,
            target_captions,
            ignore_index=self.vocab[TOKEN_PADDING],
            reduction=reduction,
        )

    def loss_rl(
        self,
        sequences,
        target_captions,
        logits,
        entropies,
        sequence_lengths,
        vocab,
        entropy_coeff,
        length_cost,
    ):
        sequences_decoded = [decode_caption(sequence, vocab) for sequence in sequences]

        references_decoded = []
        for target_captions_image in target_captions:
            references_decoded.append(
                [decode_caption(c, vocab) for c in target_captions_image]
            )

        loss = - torch.tensor(
            corpus_bleu(references_decoded, sequences_decoded), device=device
        )

        # TODO: check whether this step is superfluous
        # # the log prob/ entropy of the choices made by S before and including the eos symbol
        effective_entropy = torch.zeros(entropies.shape[0])
        effective_log_prob = torch.zeros(logits.shape[0])

        for i in range(max(sequence_lengths)):
            not_eosed = (i < sequence_lengths).float()
            effective_entropy += entropies[:, i] * not_eosed
            effective_log_prob += logits[:, i] * not_eosed
        effective_entropy = effective_entropy / sequence_lengths.float()

        weighted_entropy = effective_entropy.mean() * entropy_coeff

        length_loss = sequence_lengths.float() * length_cost

        return loss, length_loss, effective_log_prob, weighted_entropy

    def beam_search(
        self,
        encoder_output,
        beam_size,
        store_alphas=False,
        store_beam=False,
        print_beam=False,
    ):
        """Generate and return the top k sequences using beam search."""

        current_beam_width = beam_size

        enc_image_size = encoder_output.size(1)
        encoder_dim = encoder_output.size()[-1]

        # Flatten encoding
        encoder_output = encoder_output.view(1, -1, encoder_dim)

        # We'll treat the problem as having a batch size of k
        encoder_output = encoder_output.expand(
            beam_size, encoder_output.size(1), encoder_dim
        )

        # Tensor to store top k sequences; now they're just <start>
        top_k_sequences = torch.full(
            (beam_size, 1), self.vocab[TOKEN_START], dtype=torch.int64, device=device
        )

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(beam_size, device=device)

        if store_alphas:
            # Tensor to store top k sequences' alphas; now they're just 1s
            seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(
                device
            )

        # Lists to store completed sequences, scores, and alphas and the full decoding beam
        complete_seqs = []
        complete_seqs_alpha = []
        complete_seqs_scores = []
        beam = []

        # Initialize hidden states
        states = self.init_hidden_states(encoder_output)

        # Start decoding
        for step in range(0, self.max_caption_length - 1):
            prev_words = top_k_sequences[:, step]

            if step == 0:
                prev_words_embedded = self.lstm_input_first_timestep(
                    beam_size, encoder_output
                )
            else:
                prev_words_embedded = self.word_embedding(prev_words)

            predictions, states, alpha = self.forward_step(
                encoder_output, prev_words_embedded, states
            )
            scores = F.log_softmax(predictions, dim=1)

            # Add the new scores
            scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

            # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
            # different sequences, we should only look at one branch
            if step == 0:
                scores = scores[0]

            # Find the top k of the flattened scores
            top_k_scores, top_k_words = scores.view(-1).topk(
                current_beam_width, 0, largest=True, sorted=True
            )

            # Convert flattened indices to actual indices of scores
            prev_seq_inds = top_k_words / self.vocab_size  # (k)
            next_words = top_k_words % self.vocab_size  # (k)

            # Add new words to sequences
            top_k_sequences = torch.cat(
                (top_k_sequences[prev_seq_inds], next_words.unsqueeze(1)), dim=1
            )

            if print_beam:
                print_current_beam(top_k_sequences, top_k_scores, self.vocab)
            if store_beam:
                beam.append(top_k_sequences)

            # Store the new alphas
            if store_alphas:
                alpha = alpha.view(-1, enc_image_size, enc_image_size)
                seqs_alpha = torch.cat(
                    (seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)),
                    dim=1,
                )

            # Check for complete and incomplete sequences (based on the <end> token)
            incomplete_inds = (
                torch.nonzero(next_words != self.vocab[TOKEN_END]).view(-1).tolist()
            )
            complete_inds = (
                torch.nonzero(next_words == self.vocab[TOKEN_END]).view(-1).tolist()
            )

            # Set aside complete sequences and reduce beam size accordingly
            if len(complete_inds) > 0:
                complete_seqs.extend(top_k_sequences[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                if store_alphas:
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())

            # Stop if k captions have been completely generated
            current_beam_width = len(incomplete_inds)
            if current_beam_width == 0:
                break

            # Proceed with incomplete sequences
            top_k_sequences = top_k_sequences[incomplete_inds]
            for i in range(len(states)):
                states[i] = states[i][prev_seq_inds[incomplete_inds]]
            encoder_output = encoder_output[prev_seq_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds]
            if store_alphas:
                seqs_alpha = seqs_alpha[incomplete_inds]

        if len(complete_seqs) < beam_size:
            complete_seqs.extend(top_k_sequences.tolist())
            complete_seqs_scores.extend(top_k_scores)
            if store_alphas:
                complete_seqs_alpha.extend(seqs_alpha)

        sorted_sequences = [
            sequence
            for _, sequence in sorted(
                zip(complete_seqs_scores, complete_seqs), reverse=True
            )
        ]
        sorted_alphas = None
        if store_alphas:
            sorted_alphas = [
                alpha
                for _, alpha in sorted(
                    zip(complete_seqs_scores, complete_seqs_alpha), reverse=True
                )
            ]
        return sorted_sequences, sorted_alphas, beam

    def decode(self, images, num_samples, top_p=0.9, print_beam=False):
        """Generate and return the top k sequences using nucleus sampling."""
        encoder_output = self.image_encoder(images)

        current_beam_width = num_samples

        encoder_dim = encoder_output.size()[-1]

        # Flatten encoding
        encoder_output = encoder_output.view(1, -1, encoder_dim)

        # We'll treat the problem as having a batch size of k
        encoder_output = encoder_output.expand(
            num_samples, encoder_output.size(1), encoder_dim
        )

        # Tensor to store top k sequences; now they're just <start>
        top_k_sequences = torch.full(
            (num_samples, 1), self.vocab[TOKEN_START], dtype=torch.int64, device=device
        )

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(num_samples, device=device)

        # Lists to store completed sequences, scores, and alphas and the full decoding beam
        complete_seqs = []
        complete_seqs_scores = []

        # Initialize hidden states
        states = self.init_hidden_states(encoder_output)

        # Start decoding
        for step in range(0, self.max_caption_length - 1):
            prev_words = top_k_sequences[:, step]

            if step == 0:
                prev_words_embedded = self.lstm_input_first_timestep(
                    num_samples, encoder_output
                )
            else:
                prev_words_embedded = self.word_embedding(prev_words)

            predictions, states, alpha = self.forward_step(
                encoder_output, prev_words_embedded, states
            )
            scores = F.log_softmax(predictions, dim=1)

            sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            top_k_scores = torch.zeros(
                current_beam_width, dtype=torch.float, device=device
            )
            top_k_words = torch.zeros(
                current_beam_width, dtype=torch.long, device=device
            )

            for i in range(0, current_beam_width):
                scores[i][sorted_indices[i][sorted_indices_to_remove[i]]] = -float(
                    "inf"
                )

                # Sample from the scores
                top_k_words[i] = torch.multinomial(torch.softmax(scores[i], -1), 1)
                top_k_scores[i] = scores[i][top_k_words[i]]

            # Add new words to sequences
            top_k_sequences = torch.cat(
                (top_k_sequences, top_k_words.unsqueeze(1)), dim=1
            )

            if print_beam:
                print_current_beam(top_k_sequences, top_k_scores, self.vocab)

            # Check for complete and incomplete sequences (based on the <end> token)
            incomplete_inds = (
                torch.nonzero(top_k_words != self.vocab[TOKEN_END]).view(-1).tolist()
            )
            complete_inds = (
                torch.nonzero(top_k_words == self.vocab[TOKEN_END]).view(-1).tolist()
            )

            # Set aside complete sequences and reduce beam size accordingly
            if len(complete_inds) > 0:
                complete_seqs.extend(top_k_sequences[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            # Stop if k captions have been completely generated
            current_beam_width = len(incomplete_inds)
            if current_beam_width == 0:
                break

            # Proceed with incomplete sequences
            top_k_sequences = top_k_sequences[incomplete_inds]
            for i in range(len(states)):
                states[i] = states[i][incomplete_inds]
            encoder_output = encoder_output[incomplete_inds]
            top_k_scores = top_k_scores[incomplete_inds]

        if len(complete_seqs) < num_samples:
            complete_seqs.extend(top_k_sequences.tolist())
            complete_seqs_scores.extend(top_k_scores)

        sorted_sequences = [
            sequence
            for _, sequence in sorted(
                zip(complete_seqs_scores, complete_seqs), reverse=True
            )
        ]
        return sorted_sequences, None, None

    def decode_sampling(self, images):
        """Generate and return sampled sequences and probability scores for RL."""
        encoder_output = self.image_encoder(images)

        batch_size = images.shape[0]

        encoder_dim = encoder_output.size()[-1]

        # Flatten encoding
        encoder_output = encoder_output.view(batch_size, -1, encoder_dim)

        decode_lengths = torch.full(
            (batch_size,), self.max_caption_length, dtype=torch.int64, device=device,
        )

        # Tensor to store sequences
        sequences = torch.zeros(
            (batch_size, max(decode_lengths)), device=device, dtype=torch.int64,
        )

        # Tensor to store entropies
        entropies = torch.zeros(
            (batch_size, max(decode_lengths),), device=device, dtype=torch.float,
        )

        # Tensor to store sequence logits
        logits = torch.zeros(
            (batch_size, max(decode_lengths),), device=device, dtype=torch.float,
        )

        # Initialize hidden states
        states = self.init_hidden_states(encoder_output)

        # Initialize next words with SOS tokens
        next_words = torch.full(
            (batch_size,), self.vocab[TOKEN_START], dtype=torch.int64, device=device
        )

        # Start decoding
        for step in range(0, self.max_caption_length - 1):
            prev_words = next_words

            if step == 0:
                prev_words_embedded = self.lstm_input_first_timestep(
                    batch_size, encoder_output
                )
            else:
                prev_words_embedded = self.word_embedding(prev_words)

                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = (
                    torch.nonzero(prev_words == self.vocab[TOKEN_END]).view(-1).tolist()
                )

                # Update the decode lengths accordingly
                decode_lengths[ind_end_token] = torch.min(
                    decode_lengths[ind_end_token],
                    torch.full_like(decode_lengths[ind_end_token], step, device=device),
                )

            # Check if all sequences are finished:
            indices_incomplete_sequences = torch.nonzero(decode_lengths > step).view(-1)
            if len(indices_incomplete_sequences) == 0:
                break

            predictions, states, alpha = self.forward_step(
                encoder_output, prev_words_embedded, states
            )
            scores = F.log_softmax(predictions, dim=1)

            distr = Categorical(logits=scores)

            next_words = distr.sample()

            # Add new words to sequences, update entropies and logits
            sequences[indices_incomplete_sequences, step + 1] = next_words[indices_incomplete_sequences]
            entropies[indices_incomplete_sequences, step + 1] = distr.entropy()[indices_incomplete_sequences]
            logits[indices_incomplete_sequences, step + 1] = distr.log_prob(next_words)[indices_incomplete_sequences]

        return sequences, logits, entropies, decode_lengths

    def perplexity(self, images, captions, caption_lengths):
        """Return perplexities of captions given images."""

        scores, decode_lengths, alphas = self.forward(images, captions, caption_lengths)

        loss = self.loss(scores, captions, caption_lengths, alphas, reduction="none")

        perplexities = torch.exp(loss)

        # sum up cross entropies of all words
        perplexities = perplexities.sum(dim=1)

        return perplexities


def print_current_beam(top_k_sequences, top_k_scores, vocab):
    print("\n")
    for sequence, score in zip(top_k_sequences, top_k_scores):
        print(
            "{} \t\t\t\t Score: {}".format(
                decode_caption(sequence.cpu().numpy(), vocab), score
            )
        )
