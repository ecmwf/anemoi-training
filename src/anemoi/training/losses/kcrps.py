from functools import cached_property
from typing import Optional

import torch
from torch import nn

import einops 

import logging
LOGGER = logging.getLogger(__name__)

class KernelCRPS(nn.Module):
    """Area-weighted kernel CRPS loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: Optional[torch.Tensor],
        fair: bool = True,
        implementation: str = "low_mem",
        ignore_nans: Optional[bool] = False,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Args:
        node_weights : torch.Tensor
            Weights by area
        feature_weights : Optional[torch.Tensor], optional
            Loss weighting by feature
        fair: calculate a "fair" (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__()

        self.fair = fair

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)

        self.implementation = implementation

        self._kernel_crps_impl = {"low_mem": self._kernel_crps_low_mem, "vectorized": self._kernel_crps_vectorized}

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._kernel_crps_impl[self.implementation](preds, targets)

    def _kernel_crps_vectorized(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Args:
            preds: predicted ensemble, shape (batch_size, ens_size, latlon, n_vars)
            targets: ground truth, shape (batch_size, latlon, n_vars)

        Returns:
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """

        ens_size = preds.shape[2]
        mae = torch.mean(torch.abs(targets.unsqueeze(2) - preds), dim=2)

        if ens_size == 1:
            return mae
        
        # Ensemble variance term
        coef = -0.5 / (ens_size * (ens_size - 1)) if self.fair else -0.5 / (ens_size**2)
        pairwise_diffs = torch.abs(preds.unsqueeze(1) - preds.unsqueeze(2))
        ens_var = pairwise_diffs.sum(dim=(1, 2)) * coef
        
        return mae + ens_var

    def _kernel_crps_low_mem(self, preds: torch.Tensor, targets: torch.Tensor, fair: bool = True) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Args:
            preds: predicted ensemble, shape (batch_size, ens_size, latlon, n_vars)
            targets: ground truth, shape (batch_size, latlon, n_vars)
            fair: unbiased ensemble variance calculation
        Returns:
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """
        
        preds = einops.rearrange(preds, "b t e l n -> b t l n e")

        ens_size = preds.shape[-1]
        mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

        if ens_size == 1:
            return mae

        coef = -1.0 / (ens_size * (ens_size - 1)) if fair else -1.0 / (ens_size**2)

        ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
        for i in range(ens_size):  # loop version to reduce memory usage
            ens_var += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)
        ens_var = coef * ens_var

        return mae + ens_var

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_scaling: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates the area-weighted kernel CRPS loss.

        Args:
            pred: predicted ensemble, shape (batch_size, ens_size, n_vars, latlon)
            target: ground truth, shape (batch_size, n_vars, latlon)
            squash: bool, optional
                Reduce the spatial and feature dimensions
            feature_scaling: bool, optional
                Scale the loss by the feature weights
        feature_indices: indices of the features to scale the loss by

        Returns:
            Weighted kernel CRPS loss
        """
        # Apply feature scaling
        if feature_scaling:
            if feature_indices is None:
                pred = pred * self.feature_weights / self.feature_weights.numel()
                target = target * self.feature_weights / self.feature_weights.numel()
            else:
                pred = pred * self.feature_weights[..., feature_indices] / self.feature_weights.numel()
                target = target * self.feature_weights[..., feature_indices] / self.feature_weights.numel()

        # Calculate kernel CRPS
        kcrps = self._kernel_crps(pred, target)

        # Apply node (spatial) weights
        kcrps *= (self.node_weights / self.sum_function(self.node_weights))

        # Squash (reduce spatial and feature dimensions)
        if squash:
            kcrps = kcrps.sum( dim=(-2,-1) ) # (batch_size, timestep)

        return kcrps.mean(dim=0) # (timestep, latlon, nvar)
