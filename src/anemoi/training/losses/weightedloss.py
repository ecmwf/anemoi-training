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
from abc import ABC
from abc import abstractmethod
from functools import cached_property

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class WeightedLoss(nn.Module, ABC):
    """Latitude-weighted general loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
        ignore_nans: bool = False,
    ) -> None:
        """Latitude- and feature_weights Loss.

        Exposes:
        - self.avg_function: torch.nanmean or torch.mean
        - self.sum_function: torch.nansum or torch.sum
        depending on the value of `ignore_nans`

        Registers:
        - self.node_weights: torch.Tensor of shape (N, )
        - self.feature_weights: torch.Tensor if given

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

        self.register_buffer("node_weights", node_weights, persistent=True)
        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights, persistent=True)

    def scale_by_feature_weights(
        self,
        error: torch.Tensor,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Scale the error tensor by the feature_weights.

        Parameters
        ----------
        error : torch.Tensor
            Error tensor, shape (bs, (optional_ensemble), lat*lon, n_outputs)
        feature_indices:
            feature indices (relative to full model output) of the features passed in pred and target
        feature_scale:
            If True, scale the loss by the feature_weights, otherwise return the error tensor

        Returns
        -------
        torch.Tensor
            Scaled error tensor
        """
        # Use feature_weights if available and feature_scale is True
        if not feature_scale or not hasattr(self, "feature_weights"):
            return error

        if feature_indices is None:
            return error * self.feature_weights
        return error * self.feature_weights[..., feature_indices]

    def scale_by_node_weights(self, error: torch.Tensor, squash: bool = True) -> torch.Tensor:
        """
        Scale the error tensor by the node_weights.

        Parameters
        ----------
        error : torch.Tensor
            Error tensor, shape (bs, (optional_ensemble), lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True

        Returns
        -------
        torch.Tensor
            Scaled error tensor
        """
        # Squash by last dimension
        if squash:
            error = self.avg_function(error, dim=-1)
            # Weight by area
            error *= self.node_weights.expand_as(error)
            error /= self.sum_function(self.node_weights.expand_as(error))
            return self.sum_function(error)

        # Weight by area, due to weighting construction is analagous to a mean
        error *= self.node_weights[..., None].expand_as(error)
        # keep last dimension (variables) when summing weights
        error /= self.sum_function(self.node_weights[..., None].expand_as(error), dim=(0, 1, 2))
        return self.sum_function(error, dim=(0, 1, 2))

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculates the lat-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, (optional_ensemble), lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, (optional_ensemble), lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        feature_indices:
            feature indices (relative to full model output) of the features passed in pred and target
        feature_scale:
            If True, scale the loss by the feature_weights

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        # If pred is 4D, average over ensemble dimension
        if pred.ndim == 4:
            pred = pred.mean(dim=1)

        out = pred - target

        out = self.scale_by_feature_weights(out, feature_indices, feature_scale)
        return self.scale_by_node_weights(out, squash)

    @cached_property
    def name(self) -> str:
        """Used for logging identification purposes."""
        return self.__class__.__name__.lower()
