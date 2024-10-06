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

import torch
from torch import nn

from functools import cached_property
from typing import Optional

import torch
from torch import nn
from .mixins import TargetEachEnsIndepMixin
LOGGER = logging.getLogger(__name__)
from typing import Union

class WeightedMAELoss(TargetEachEnsIndepMixin, nn.Module):
    """Latitude-weighted MAE loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor,
        ignore_nans: bool | None = False,
        target_each_ens_indep: bool = False,
    ) -> None:
        """Latitude- and feature-weighted MAE Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        feature_weights : torch.Tensor of shape (N, )
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[..., None], persistent=True)

        self.register_buffer("feature_weights", feature_weights, persistent=True)
        self.target_each_ens_indep = target_each_ens_indep
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: Union[bool, tuple] = True,
        feature_scale: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted MAE loss.

        Args:
            pred: Prediction tensor, shape (bs, nens_input, timesteps, lat*lon, n_outputs)
            target: Target tensor, shape (bs, nens_target, timesteps, lat*lon, n_outputs)
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
        out = out * (self.node_weights / self.sum_function(self.node_weights))

        # Reduce over ensemble dimension
        out = out.mean(1)

        # Squash - reduce spatial and feature dimensions
        if squash:
            out = self.sum_function(out, axis=squash if isinstance(squash, tuple) else (-3, -2, -1))

        return self.avg_function(out, axis=(0))  # (timesteps) or (timesteps, latlon, nvars)

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss for logging."""
        return "mae"
