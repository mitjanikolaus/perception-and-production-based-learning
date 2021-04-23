import math
import os
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd

from eval_semantics import eval_semantics_score
from utils import CHECKPOINT_NAME_SENDER, CHECKPOINT_NAME_RECEIVER, decode_caption

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass
from torch.utils.data import DataLoader

from egg.core import Trainer, Callback, Interaction, move_to, get_opts


class VisualRefTrainer(Trainer):
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        out_checkpoints_dir: str,
        train_data: DataLoader,
        semantics_eval_loaders,
        vocab,
        validation_data: Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
        print_sample_interactions: bool = False,
        print_sample_interactions_images: bool = False,
    ):
        super(VisualRefTrainer, self).__init__(
            game,
            optimizer,
            train_data,
            validation_data,
            device,
            callbacks,
            grad_norm,
            aggregate_interaction_logs,
        )
        common_opts = get_opts()
        self.eval_frequency = common_opts.eval_frequency

        self.best_val_loss = math.inf

        self.out_checkpoints_dir = out_checkpoints_dir

        self.semantics_eval_loaders = semantics_eval_loaders

        self.accuracies_over_time = []

        self.vocab = vocab

        self.print_sample_interactions = print_sample_interactions
        self.print_sample_interactions_images = print_sample_interactions_images

    def save_models(self):
        torch.save(
            {
                "model_state_dict": self.game.receiver.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.best_val_loss,
            },
            os.path.join(self.out_checkpoints_dir, CHECKPOINT_NAME_RECEIVER),
        )
        torch.save(
            {
                "model_state_dict": self.game.sender.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.best_val_loss,
            },
            os.path.join(self.out_checkpoints_dir, CHECKPOINT_NAME_SENDER),
        )

    def print_interactions(self, interaction_logs, show_images, num_interactions=5):
        for z in range(num_interactions):
            message = decode_caption(interaction_logs.message[z], self.vocab)
            target_position = interaction_logs.labels[z]
            receiver_guess = torch.argmax(interaction_logs.receiver_output[z])
            target_image_id, distractor_image_id = interaction_logs.sender_input[z]

            if show_images:
                # plot the two images side-by-side
                target_image = self.train_dataset.get_image_features(
                    int(target_image_id), channels_first=False, normalize=False
                )
                distractor_image = self.train_dataset.get_image_features(
                    int(distractor_image_id), channels_first=False, normalize=False
                )
                image = torch.cat([target_image, distractor_image], dim=1).cpu().numpy()

                plt.title(
                    f"Left: Target, Right: Distractor"
                    f"\nReceiver guess correct: {target_position == receiver_guess}"
                    f"\nMessage: {message}"
                )
                plt.imshow(image)
                plt.show()
            else:
                print(
                    f"Target image ID: {target_image_id} | Distractor image ID: {distractor_image_id} | "
                    f"Success: {target_position == receiver_guess} | Message: {message}"
                )

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(self.train_data):
            if batch_id % self.eval_frequency == 0:
                val_loss, val_interactions = self.eval()
                val_acc = val_interactions.aux["acc"].mean().item()
                print(
                    f"Batch {batch_id} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f}\n"
                )
                accuracies = {"batch_id": batch_id, "val_loss": val_loss, "val_acc": val_acc}

                for name, semantic_images_loader in self.semantics_eval_loaders.items():
                    acc = eval_semantics_score(
                        self.game.sender, semantic_images_loader, self.vocab
                    )
                    print(f"Accuracy for {name}: {acc:.3f}")
                    accuracies[name] = acc

                self.accuracies_over_time.append(accuracies)
                pd.DataFrame(self.accuracies_over_time).to_csv(
                    os.path.join(
                        self.out_checkpoints_dir,
                        CHECKPOINT_NAME_SENDER.replace(".pt", "_accuracies.csv"),
                    )
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_models()

                if (
                        self.print_sample_interactions
                        or self.print_sample_interactions_images
                ):
                    self.print_interactions(
                        val_interactions,
                        show_images=self.print_sample_interactions_images,
                    )

            self.game.train()

            batch = move_to(batch, self.device)

            context = autocast() if self.scaler else nullcontext()
            with context:
                optimized_loss, interaction = self.game(*batch)

                if self.update_freq > 1:
                    # throughout EGG, we minimize _mean_ loss, not sum
                    # hence, we need to account for that when aggregating grads
                    optimized_loss = optimized_loss / self.update_freq

            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            if (
                self.distributed_context.is_distributed
                and self.aggregate_interaction_logs
            ):
                interaction = Interaction.gather_distributed_interactions(interaction)
            interaction = interaction.to("cpu")

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)


        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)
        return mean_loss, full_interaction
