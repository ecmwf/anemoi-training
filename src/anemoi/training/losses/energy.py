import os
import types
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn, Tensor
from anemoi.training.losses.utils import process_file
import logging

LOGGER = logging.getLogger(__name__)

class EnergyScore(nn.Module):
    """EnergyScore loss for ensemble forecasts.

    Attributes
    ----------
    node_weights : torch.Tensor
        Weights for different areas in the loss calculation.
    feature_weights : Optional[torch.Tensor]
        Scaling factors applied to the loss for each feature.
    group_on_dim : int
        Dimension to group on for the loss calculation (-1 for feature, -2 for spatial, -3 for temporal).
    power : torch.Tensor or int
        Power exponent for the energy loss calculation.
    fair : bool
        If True, apply a fair version of the energy score, else use the standard version.
    ignore_nans : Optional[bool]
        Flag to determine whether to ignore NaN values during calculations.
    logging : str
        Logging level for the class.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: Optional[torch.Tensor],
        group_on_dim: int = -1,
        power: Union[torch.Tensor, int] = 1.0,
        p_norm: Union[torch.Tensor, int] = 2,
        fair: bool = True,
        ignore_nans: Optional[bool] = False,
        logging: str = "INFO",
        **kwargs,
    ) -> None:
        super().__init__()
        LOGGER.setLevel(logging)

        assert group_on_dim in [-2, -1], f"Invalid group dimension: {group_on_dim}"

        self.group_on_dim = group_on_dim
        self.fair = fair

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        area_exp = power if self.group_on_dim == -2 else None
        feature_exp = power if self.group_on_dim == -1 else None

        # Node weights normalization
        self.register_buffer("node_weights", node_weights[..., None], persistent=False)
        self.register_buffer(
            "node_weights_normalized",
            self.node_weights / torch.sum(self.node_weights),
            persistent=False,
        )

        if area_exp is not None:
            self.node_weights_normalized = torch.pow(self.node_weights_normalized, 1 / area_exp)

        # Feature weights normalization
        assert feature_weights is not None, "Feature weights must be provided."
        self.register_buffer("feature_weights", feature_weights, persistent=False)
        self.register_buffer(
            "feature_weights_normalized",
            self.feature_weights,
            persistent=False,
        )

        if feature_exp is not None:
            self.feature_weights_normalized = torch.pow(self.feature_weights_normalized, 1 / feature_exp)

        # Power and p_norm initialization
        self.register_buffer("power", torch.as_tensor(power), persistent=False)
        if self.power.ndim == 0:
            self.power = self.power.unsqueeze(0)
        self.p_norm = p_norm

        # Adjust node weights for feature grouping
        if self.group_on_dim == -1:
            self.node_weights_normalized = self.node_weights_normalized[..., 0]

        # Dictionary of forward functions based on group dimension
        self.forward_funcs = {
            -1: self._forward_feature,
            -2: self._forward_spatial,
            -3: self._forward_temporal,
        }

    def _calc_energy_score(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate energy score.

        Args:
            preds (torch.Tensor): Forecast realizations (bs, nens, timesteps, latlon, nvar).
            target (torch.Tensor): Ground truth (bs, timesteps, latlon, nvar).

        Returns:
            torch.Tensor: Energy score loss component for gradient descent (bs, latlon/nvar).
        """
        # Precision component (prediction vs target)
        pairwise_diff_predictor = torch.linalg.vector_norm(preds - target.unsqueeze(2), dim=self.group_on_dim, ord=self.p_norm).pow(self.power)

        # Mean over ensemble
        precision_score = pairwise_diff_predictor.mean(dim=1)  # shape (bs, timesptes/latlon/nvar)

        # Spread component (difference between co-predictors)
        pairwise_diff_copredictors = torch.linalg.vector_norm(
            preds.unsqueeze(1) - preds.unsqueeze(2),
            dim=self.group_on_dim,
            ord=self.p_norm,
        ).pow(self.power)

        if self.fair:
            nens = preds.shape[1]
            coeff = -0.5* (1 / (nens * (nens - 1)))
        else:
            coeff = =0.5  * 1/(nens**2)
        
        spread_score = pairwise_diff_copredictors.sum(dim=(1, 2)) * coeff

        # Energy score is the difference between precision and spread
        return precision_score - spread_score

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Forward pass for energy score calculation."""
        return self.forward_funcs[self.group_on_dim](preds, target, squash, feature_indices, feature_scale)

    def _forward_spatial(self, preds: torch.Tensor, target: torch.Tensor, squash: bool, feature_indices: Optional[torch.Tensor], feature_scale: bool) -> torch.Tensor:
        """Forward pass for spatial grouping (group_on_dim = -2)."""
        preds = preds * self.node_weights_normalized
        target = target * self.node_weights_normalized

        energy_score = self._calc_energy_score(preds, target)

        if feature_scale:
            energy_score = energy_score * self._scale_feature(energy_score, feature_indices)

        return self._reduce_output(energy_score, squash)

    def _forward_feature(self, preds: torch.Tensor, target: torch.Tensor, squash: bool, feature_indices: Optional[torch.Tensor], feature_scale: bool) -> torch.Tensor:
        """Forward pass for feature grouping (group_on_dim = -1)."""
        if feature_scale:
            feature_scale = self._scale_feature(None, feature_indices)
            preds, target = preds * feature_scale, target * feature_scale

        energy_score = self._calc_energy_score(preds, target)
        energy_score = energy_score * self.node_weights_normalized[:, 0]

        return self._reduce_output(energy_score, squash)

    def _forward_temporal(self, preds: torch.Tensor, target: torch.Tensor, squash: bool, feature_indices: Optional[torch.Tensor], feature_scale: bool) -> torch.Tensor:
            """Forward pass for temporal grouping (group_on_dim = -3)."""
            # No node weights applied since this is temporal
            energy_score = self._calc_energy_score(preds, target)

            energy_score = energy_score * self.node_weights_normalized
            if feature_scale:
                energy_score = energy_score * self._scale_feature(feature_indices)
            
            return self._reduce_output(energy_score, squash)

    def _scale_feature(self, feature_indices: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply scaling for feature dimensions."""
        return self.feature_weights_normalized if feature_indices is None else self.feature_weights_normalized[..., feature_indices]


    def _reduce_output(self, energy_score: torch.Tensor, squash: bool) -> torch.Tensor:
        """Reduce the output to a single value or return the tensor."""
        if squash:
            energy_score = energy_score.sum(dim=(1,2))
    
        return energy_score.mean(0)

    @cached_property
    def log_name(self) -> str:
        """Generate a log name based on parameters."""
        power_str = f"p{format(self.power.item(), '.2g')}" if torch.is_tensor(self.power) else f"p{format(self.power, '.2g')}"
        fair_str = "f" if self.fair else ""
        pnorm_str = f"_pnorm{self.p_norm}" if self.p_norm != 2 else ""
        return f"{fair_str}energy_d{self.group_on_dim}_{power_str}{pnorm_str}"
