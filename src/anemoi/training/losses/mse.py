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

LOGGER = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """Latitude-weighted MSE loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        data_variances: torch.Tensor | None = None,
        ignore_nans: bool | None = False,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted MSE Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        data_variances : Optional[torch.Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("weights", node_weights, persistent=True)
        if data_variances is not None:
            self.register_buffer("ivar", data_variances, persistent=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
    ) -> torch.Tensor:
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
            out = self.avg_function(out, dim=-1)
            # Weight by area
            out *= self.weights.expand_as(out)
            out /= self.sum_function(self.weights.expand_as(out))
            return self.sum_function(out)

        # Weight by area
        out *= self.weights[..., None].expand_as(out)
        # keep last dimension (variables) when summing weights
        out /= self.sum_function(self.weights[..., None].expand_as(out), axis=(0, 1, 2))
        return self.sum_function(out, axis=(0, 1, 2))


class WeightedMSELossStretchedGrid(nn.Module):
    """Latitude-weighted MSE loss, calculated only within or outside the limited area.
    Further, the loss can be computed for the specified region (default),
    or as the contribution to the overall loss.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        mask: torch.Tensor,
        inside_LAM: bool = True,
        wmse_contribution: bool = False,
        data_variances: torch.Tensor | None = None,
        ignore_nans: bool | None = False,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted MSE Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        mask: torch.Tensor
            the mask marking the indices of the regional data points (bool)
        inside_LAM: bool
            compute the loss inside or outside the limited area, by default inside (True)
        wmse_contribution: bool
            compute loss as the contribution to the overall MSE, by default False
        data_variances : Optional[torch.Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.inside_LAM = inside_LAM
        self.wmse_contribution = wmse_contribution
        self.register_buffer("weights", node_weights, persistent=True)
        self.register_buffer("weights_inside_LAM", node_weights[mask], persistent=True)
        self.register_buffer("weights_outside_LAM", node_weights[~mask], persistent=True)
        self.register_buffer("mask", mask, persistent=True)
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
        full_out_dims = pred[:, :, :, 0]

        if self.inside_LAM:
            pred = pred[:, :, self.mask]
            target = target[:, :, self.mask]
            weights_selected = self.weights_inside_LAM
        else:
            pred = pred[:, :, ~self.mask]
            target = target[:, :, ~self.mask]
            weights_selected = self.weights_outside_LAM

        out = torch.square(pred - target)

        # Use variances if available
        if hasattr(self, "ivar"):
            out *= self.ivar

        # Squash by last dimension
        if squash:
            out = self.avg_function(out, dim=-1)
            # Weight by area
            out = out * weights_selected.expand_as(out)
            if self.wmse_contribution:
                out /= self.sum_function(self.weights.expand_as(full_out_dims))
            else:
                out /= self.sum_function(weights_selected.expand_as(out))
            return self.sum_function(out)

        # Weight by area
        out = out * weights_selected[..., None].expand_as(out)
        if self.wmse_contribution:
            out /= self.sum_function(self.weights[..., None].expand_as(full_out_dims))
        else:
            out /= self.sum_function(weights_selected[..., None].expand_as(out))
        return self.sum_function(out, axis=(0, 1, 2))
