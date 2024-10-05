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

from anemoi.training.losses.weightedloss import WeightedLoss

LOGGER = logging.getLogger(__name__)


class WeightedMAELoss(WeightedLoss):
    """Latitude-weighted MAE loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Latitude- and feature weighted MAE Loss.

        Also known as the Weighted L1 loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        feature_weights : Optional[torch.Tensor], optional
            precomputed, per-variable weights, by default None
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__(node_weights=node_weights, feature_weights=feature_weights, ignore_nans=ignore_nans, **kwargs)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculates the lat-weighted MAE loss.

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
            Weighted MAE loss
        """
        # If pred is 4D, average over ensemble dimension
        if pred.ndim == 4:
            pred = pred.mean(dim=1)

        out = torch.abs(pred - target)

        out = self.scale_by_feature_weights(out, feature_indices, feature_scale)
        return self.scale_by_node_weights(out, squash)

    @cached_property
    def name(self) -> str:
        return "wmae"
