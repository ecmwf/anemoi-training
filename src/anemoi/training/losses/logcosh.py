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

import numpy as np
import torch

from anemoi.training.losses.weightedloss import BaseWeightedLoss

LOGGER = logging.getLogger(__name__)


class WeightedLogCoshLoss(BaseWeightedLoss):
    """Node-weighted LogCosh loss."""

    name = "wlogcosh"

    def __init__(
        self,
        node_weights: torch.Tensor,
        variable_scaling: torch.Tensor | None = None,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted LogCosh Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        variable_scaling : Optional[torch.Tensor], optional
            precomputed, per-variable weights, by default None
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__(
            node_weights=node_weights,
            variable_scaling=variable_scaling,
            ignore_nans=ignore_nans,
            **kwargs,
        )

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
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
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
        # Keep logcosh numerically stable
        x = pred - target
        s = torch.sign(x) * x
        p = torch.exp(-2 * s)
        out = s + torch.log1p(p) - np.log(2)

        if feature_scale:
            out = self.scale_by_variable_scaling(out, feature_indices)
        return self.scale_by_node_weights(out, squash)
