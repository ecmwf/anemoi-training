from __future__ import annotations

import logging
import hydra


from functools import cached_property

import torch
from torch import nn
from torch import Tensor
from typing import TYPE_CHECKING
from typing import Union

from anemoi.training.losses.utils import buffered_arange

LOGGER = logging.getLogger(__name__)
if TYPE_CHECKING:
    from omegaconf import DictConfig


class CompositeLoss(nn.Module):  # A composite loss that combines multiple losses
    def __init__(self, losses: torch.nn.ModuleList | list[DictConfig], loss_weights: torch.Tensor | None = None, **kwargs) -> None:
        """Composite loss that combines multiple losses.

        Args:
            losses: Either a ModuleList of loss functions or a list of configurations for Hydra to instantiate.
            loss_weights: Weights for each loss function. If None, weights are uniform.
        """
        super().__init__()

        # If losses is a list of configs, instantiate the losses via Hydra
        if isinstance(losses, list):
            _ = [hydra.utils.instantiate(cfg, **kwargs) for cfg in losses]
            losses = torch.nn.ModuleList(_)


        # If weights are provided, ensure they match the number of losses
        if loss_weights is not None:
            assert len(losses) == len(loss_weights), "Length of losses and weights must be equal"
            loss_weights = torch.as_tensor([weight / sum(loss_weights) for weight in loss_weights])
        else:
            # Set uniform weights if not provided
            loss_weights = torch.full((len(losses),), 1 / len(losses))

        # Store the losses and their respective weights
        self.losses_len = torch.as_tensor(len(losses))
        self.register_buffer("loss_weights", torch.as_tensor(loss_weights))
        self.losses = losses

    def forward(self, preds: Tensor, target: Tensor, squash: Union[bool, tuple] = True, **kwargs) -> Tensor:
        """Forward pass of the composite loss.

        Args:
            preds: Predicted values, shape (batch_size, ens_size_inp, timesteps, latlon, n_vars)
            target: Ground truth values, shape (batch_size, ens_size_target, timesteps, latlon, n_vars)
            squash: If False, return a (latlon, n_vars) tensor with the individual loss contributions;
                if True, return the (scalar) total loss.

        # When squash is False the shape of output from each individual loss may be different
        # The losses are then broadcasted to the target shape and summed up
        # The final loss is the weighted sum of the losses
        """
        if squash:
            loss_values = preds.new_zeros(len(self.losses))

            for idx in buffered_arange(self.losses_len):
                loss_values[idx] = self.losses[idx](preds, target, squash=squash, **kwargs)

            composite_loss = torch.sum(loss_values * self.loss_weights)
        else:
            loss_values = [None for _ in range(self.losses_len)]

            for idx in buffered_arange(self.losses_len):
                loss_values[idx] = self.losses[idx](preds, target, squash=squash, **kwargs)

            # Get the shape of the final two dimensions of preds
            target_shape = preds.shape[-2:]

            # Broadcast logic for multiple loss tensors (group of losses)
            for idx in buffered_arange(self.losses_len):
                loss_value = loss_values[idx]
                original_num_elements = loss_value.numel()  # Original number of elements before broadcasting

                if loss_value.shape != target_shape:
                    # Adjust the loss_value to match the target_shape
                    if loss_value.ndim != 2 and loss_value.shape[0] == target_shape[0]:
                        loss_value = loss_value.unsqueeze(1)
                    loss_value = loss_value.expand(*target_shape)

                    # Calculate the broadcast factor and divide by it to ensure loss consistency
                    broadcast_num_elements = loss_value.numel()  # Number of elements after broadcasting
                    broadcast_factor = broadcast_num_elements / original_num_elements

                    # Scale down the loss value by the broadcast factor
                    loss_value = loss_value / broadcast_factor

                loss_values[idx] = loss_value

            # Adjusted to accumulate the weighted losses
            composite_loss = sum(weight * loss_value for weight, loss_value in zip(self.loss_weights, loss_values))

        return composite_loss

    @cached_property
    def name(self) -> str:

        sublossweight_str = "_".join([f"{loss.name}-{weight:.2g}" for loss, weight in zip(self.losses, self.loss_weights.tolist())])

        return f"composite_{sublossweight_str}"
