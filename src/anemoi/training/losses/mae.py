from functools import cached_property
from typing import Optional

import torch
from torch import nn
import logging 

LOGGER = logging.getLogger(__name__)




class WeightedMAELoss(nn.Module):
    """Weighted MAE loss."""

    def __init__(
        self, node_weights: torch.Tensor, feature_weights: Optional[torch.Tensor], logging: str = "INFO", **kwargs,
    ) -> None:
        """Area-weighted + component-scaled MAE Loss.

        Args:
            node_weights: area weights
            feature_weights: weight loss components to ensure all vars contribute ~ equally to the total value
        """
        super().__init__()

        LOGGER.setLevel(logging)

        self.register_buffer("node_weights", node_weights[...,None], persistent=True)

        self.register_buffer("feature_weights", feature_weights, persistent=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_scale: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates the area-weighted MAE loss.

        Args:
            pred: Prediction tensor, shape (bs, ens, timesteps, lat*lon, n_outputs)
            target: Target tensor, shape (bs,  ens, timesteps, lat*lon, n_outputs)
            squash: if False, return a (n_outputs, 1) tensor with the individual loss contributions
                    if True, return the (scalar) total loss
            feature_scale: if True, scale the loss by the feature weights
            feature_indices: indices of the features to scale the loss by
        """
        out = torch.abs(pred - target)

        if feature_scale:
            out = (out * self.feature_weights) if feature_indices is None else (out * self.feature_weights[..., feature_indices])
            out = out / self.feature_weights.numel()

        # Scale in spatial dimension
        out *= (self.node_weights / self.sum_function(self.node_weights))

        # Squash - reduce spatial and feature dimensions
        if squash:
            out = self.sum_function(out, axis=(1,2))

        return self.avg_function(out, axis=(0)) 

    @cached_property
    def log_name(self) -> str:
        """Returns the name of the loss for logging."""
        return "mae"
