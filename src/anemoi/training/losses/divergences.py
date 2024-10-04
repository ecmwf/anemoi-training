from __future__ import annotations

import logging

import torch
from torch import nn
from torch import Tensor
from typing import Optional
from functools import cached_property
LOGGER = logging.getLogger(__name__)


class KLDivergenceLoss(nn.Module):
    """KL Divergence loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: Optional[torch.Tensor] = None,
        ignore_nans: bool | None = False,
    ) -> None:
        """Latitude- and feature-weighted KL Divergence Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function.
        feature_weights : Optional[torch.Tensor of shape (N, )], optional
            Weight of each feature in the loss function, by default None.
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False.

        """
        super().__init__()
        
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[...,None], persistent=True)
        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights, persistent=True)
        else:
            self.feature_weights = None

    def _kl_div(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """KL divergence for a zero mean and unit variance distribution."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    def forward(
        self,
        mu: Tensor,
        logvar: Tensor,
        squash: bool = True,
        feature_scaling: bool = False,
        feature_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Calculates the latitude-weighted KL Divergence loss.

        Parameters
        ----------
        mu : torch.Tensor
            Mean tensor from the VAE encoder, shape (bs, timesteps, lat*lon, dim)
        logvar : torch.Tensor
            Log-variance tensor from the VAE encoder, shape (bs, timesteps, lat*lon, dim)
        squash : bool, optional
            Reduce the spatial and feature dimensions, by default True
        feature_scaling : bool, optional
            Apply feature weights to the loss, by default True
        feature_indices: indices of the features to scale the loss by
        

        Returns
        -------
        torch.Tensor
            Weighted KL Divergence loss.
        """
        kl_loss = self._kl_div(mu, logvar)
        
        # Scale in feature dimension
        if feature_scaling:
            kl_loss = (kl_loss * self.feature_weights if feature_indices is None else kl_loss * self.feature_weights[..., feature_indices])
            kl_loss = kl_loss / self.feature_weights.numel()  # Normalize by number of features

        # Scale in spatial dimension
        kl_loss *= (self.node_weights / self.sum_function(self.node_weights))

        # Squash - reduce spatial and feature dimensions
        if squash:
            kl_loss = self.sum_function(kl_loss, axis=(-3, -2, -1))

        return self.avg_function(kl_loss, axis=(0))  # (timesteps, latlon, dim)

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss for logging."""
        return "kldiv"

class RenyiDivergenceLoss(nn.Module):
    """Rényi Divergence loss."""

    def __init__(
        self,
        alpha: float,
        node_weights: torch.Tensor,
        feature_weights: Optional[torch.Tensor] = False,
        ignore_nans: bool | None = False,
    ) -> None:
        """Latitude- and feature-weighted Rényi Divergence Loss.

        Parameters
        ----------
        alpha : float
            The order of the Rényi divergence. When alpha=1, it becomes KL Divergence.
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function.
        feature_weights : Optional[torch.Tensor of shape (N, )], optional
            Weight of each feature in the loss function, by default None.
        ignore_nans : bool, optional
            Allow NaNs in the loss and apply methods ignoring NaNs, by default False.
        """
        super().__init__()
        
        self.alpha = alpha
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[...,None], persistent=True)
        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights, persistent=True)
        else:
            self.feature_weights = None

    def _renyi_div(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Rényi divergence for order alpha."""
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return (1 / (self.alpha - 1)) * torch.log(1 + (self.alpha - 1) * kl_div)

    def forward(
        self,
        mu: Tensor,
        logvar: Tensor,
        squash: bool = True,
        feature_scaling: bool = False,
        feature_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Calculates the latitude-weighted Rényi Divergence loss.

        Parameters
        ----------
        mu : torch.Tensor
            Mean tensor, shape (bs,timesteps, lat*lon, dim)
        logvar : torch.Tensor
            Log-variance tensor, shape (bs, timesteps, lat*lon, dim)
        squash : bool, optional
            Reduce the spatial and feature dimensions, by default True
        feature_scaling : bool, optional
            Apply feature weights to the loss, by default True
        feature_indices: indices of the features to scale the loss by
        

        Returns
        -------
        torch.Tensor
            Weighted Rényi Divergence loss.
        """
        renyi_loss = self._renyi_div(mu, logvar)

        # Scale in feature dimension
        if feature_scaling:
            renyi_loss = renyi_loss * self.feature_weights if feature_indices is None else renyi_loss * self.feature_weights[..., feature_indices]
            renyi_loss = renyi_loss / self.feature_weights.numel()
        
        # Scale in spatial dimension
        renyi_loss *= (self.node_weights / self.sum_function(self.node_weights))

        # Squash - reduce spatial and feature dimensions
        if squash:
            renyi_loss = self.sum_function(renyi_loss, axis=(-3, -2, -1))
        
        return self.avg_function(renyi_loss, axis=(0))  # (latlon, dim)

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss for logging."""
        return "renyi_div"
