# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from functools import cached_property

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class WeightedLogCoshLoss(nn.Module):
    """Latitude-weighted LogCosh loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
        ignore_nans: bool | None = False,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted LogCosh Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        feature_weights : Optional[torch.Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("weights", node_weights, persistent=True)
        if feature_weights is not None:
            self.register_buffer("ivar", feature_weights, persistent=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculates the lat-weighted LogCosh loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        feature_indices:
            feature indices (relative to full model output) of the features passed in pred and target
        feature_scale:
            If True, scale the loss by the feature_weights

        Returns
        -------
        torch.Tensor
            Weighted LogCosh loss

        """
        if pred.ndim == 4:
            pred = pred.mean(dim=1)

        out = torch.log(torch.cosh(pred - target))

        # Use variances if available
        if feature_scale and hasattr(self, "feature_weights"):
            out = (
                out * self.feature_weights
                if feature_indices is None
                else out * self.feature_weights[..., feature_indices]
            )

        # Squash by last dimension
        if squash:
            out = self.avg_function(out, dim=-1)
            # Weight by area
            out *= self.weights.expand_as(out)
            out /= self.sum_function(self.weights.expand_as(out))
            return self.sum_function(out)

        # Weight by area, due to weighting construction is analagous to a mean
        out *= self.weights[..., None].expand_as(out)
        # keep last dimension (variables) when summing weights
        out /= self.sum_function(self.weights[..., None].expand_as(out), axis=(0, 1, 2))
        return self.sum_function(out, axis=(0, 1, 2))

    @cached_property
    def name(self) -> str:
        return "logcosh"
