
from __future__ import annotations

import logging

import torch
from torch import nn
from functools import cached_property
LOGGER = logging.getLogger(__name__)


class SpreadSkillLoss(nn.Module):
    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
        ignore_nans: bool = False,
    ) -> None:
        """SpreadSkill.

        Parameters
        ----------
        node_weights : torch.Tensor
            Weight of each node in the spatial domain
        feature_weights : Optional[torch.Tensor]
            Weights for each feature, by default None
        ignore_nans : bool
            Whether to ignore NaN values in the calculations, by default False
        """
        super().__init__()
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_scale: bool = True,
        feature_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates the SpreadSkill loss."""
        rmse = torch.sqrt(torch.square(preds.mean(dim=1) - target))
        ens_stdev = torch.sqrt(torch.square(preds - preds.mean(dim=1, keepdim=True)).sum(dim=1) / (preds.shape[1] - 1))

        if feature_scale:
            rmse = (rmse * self.feature_weights) if feature_indices is None else (rmse * self.feature_weights[..., feature_indices])
            ens_stdev = (ens_stdev * self.feature_weights) if feature_indices is None else (ens_stdev * self.feature_weights[..., feature_indices])

        # Scale in spatial dimension
        rmse *= (self.node_weights / self.sum_function(self.node_weights))
        ens_stdev *= (self.node_weights / self.sum_function(self.node_weights))

        if squash:
            rmse = self.sum_function(rmse, axis=(-3, -2, -1))
            ens_stdev = self.sum_function(ens_stdev, axis=(-3, -2, -1))

        return self.avg_function(ens_stdev / rmse, axis=0)

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss function."""
        beta_str = "b" + format(self.beta.item(), ".2g") if torch.is_tensor(self.beta) else "b" + format(self.beta, ".2g")
        return f"spread_skill_{beta_str}"

class SpreadLoss(nn.Module):
    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
        beta: int = 2,
        ignore_nans: bool = False,
    ) -> None:
        """Spread.

        Parameters
        ----------
        node_weights : torch.Tensor
            Weight of each node in the spatial domain
        feature_weights : Optional[torch.Tensor]
            Weights for each feature, by default None
        beta : int
            Power for the spread calculation, default is 2
        ignore_nans : bool
            Whether to ignore NaN values in the calculations, by default False
        """
        super().__init__()
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.beta = beta
        self.register_buffer("node_weights", node_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_scale: bool = True,
        feature_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates the Spread loss."""
        spread = torch.pow(
            torch.pow(preds - preds.mean(dim=1, keepdim=True), self.beta).mean(dim=1), 1 / self.beta,
        )

        if feature_scale:
            spread = (spread * self.feature_weights) if feature_indices is None else (spread * self.feature_weights[..., feature_indices])

        # Scale in spatial dimension
        spread *= (self.node_weights / self.sum_function(self.node_weights))

        if squash:
            spread = self.sum_function(spread, axis=(-3, -2, -1))

        return self.avg_function(spread, axis=0)

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss function."""
        return "spread"

class ZeroSpreadRateLoss(nn.Module):
    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
        ignore_nans: bool = False,
    ) -> None:
        """ZeroSpreadRateLoss.

        Parameters
        ----------
        node_weights : torch.Tensor
            Weight of each node in the spatial domain
        feature_weights : Optional[torch.Tensor]
            Weights for each feature, by default None
        ignore_nans : bool
            Whether to ignore NaN values in the calculations, by default False
        """
        super().__init__()
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_scale: bool = True,
        feature_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates the Zero Spread Rate loss."""
        spread = torch.square(preds - preds.mean(dim=1, keepdim=True)).mean(dim=1)
        spread_occurence = torch.where(spread == 0, 1.0, 0.0)

        if feature_scale:
            spread_occurence = (spread_occurence * self.feature_weights) if feature_indices is None else (spread_occurence * self.feature_weights[..., feature_indices])

        # Scale in spatial dimension
        spread_occurence *= (self.node_weights / self.sum_function(self.node_weights))

        if squash:
            spread_occurence = self.sum_function(spread_occurence, axis=(-3, -2, -1))

        return self.avg_function(spread_occurence, axis=0)

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss function."""
        return "zero_spread_rate"