# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """Latitude-weighted MSE loss."""

    def __init__(self, area_weights: torch.Tensor, data_variances: Optional[torch.Tensor] = None) -> None:
        """Latitude- and (inverse-)variance-weighted MSE Loss.

        Parameters
        ----------
        area_weights : torch.Tensor
            Weights by area
        data_variances : Optional[torch.Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        """
        super().__init__()

        self.register_buffer("weights", area_weights, persistent=True)
        if data_variances is not None:
            self.register_buffer("ivar", data_variances, persistent=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, squash=True) -> torch.Tensor:
        """Calculates the lat-weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True

        Returns
        -------
        torch.Tensor
            Weighted MSE loss
        """
        out = torch.square(pred - target)

        # Use variances if available
        if hasattr(self, "ivar"):
            out *= self.ivar

        # Squash by last dimension
        if squash:
            out = out.mean(dim=-1)
            out = out * self.weights.expand_as(out)
            out /= torch.sum(self.weights.expand_as(out))
            return out.sum()

        # Weight by area
        out = out * self.weights[..., None].expand_as(out)
        out /= torch.sum(self.weights[..., None].expand_as(out))
        return out.sum(axis=(0, 1, 2))
