# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from functools import cached_property
import logging

import torch
from torch import nn
from .mixins import TargetEachEnsIndepMixin
LOGGER = logging.getLogger(__name__)
from typing import Union

# TODO(rilwan-ade): make parent loss calss that holds the common methods avg_function and sum_function

class WeightedMSELoss(TargetEachEnsIndepMixin,nn.Module):
    """Latitude-weighted MSE loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor,
        ignore_nans: bool | None = False,
        target_each_ens_indep: bool = False,
    ) -> None:
        """Latitude- and feature-weighted MSE Loss.

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
        feature_indices: torch.Tensor | None = None,
        **kwargs,
        ) -> torch.Tensor:
        """Calculates the lat-weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ens, (timesteps), lat*lon, n_mseputs)
        target : torch.Tensor
            Target tensor, shape (bs, ens, (timesteps), lat*lon, n_mseputs)
        squash : bool, optional
            Reduce the spatial and feature dimensions
        feature_scaling : bool, optional
            Scale the loss by the feature weights
        feature_indices: indices of the features to scale the loss by

        Returns
        -------
        torch.Tensor
            Weighted MSE loss

        """
        mse = self.calc(pred, target) 

        mse = self.scale(mse, feature_scale, feature_indices, squash)

        return mse 

    def calc(pred, target):
        
        mse = torch.square(pred - target)


        return mse

    def scale(self, loss: torch.Tensor, feature_scale: bool = True, feature_indices: torch.Tensor | None = None, squash: Union[bool, tuple] = True):
        # Scale in feature dimension
        
        if feature_scale:
            mse = (mse * self.feature_weights) if feature_indices is None else (mse * self.feature_weights[..., feature_indices])
            mse = mse / self.feature_weights.numel()
            # Normalize by number of features

        # Scale in spatial dimension
        mse *= (self.node_weights / self.sum_function(self.node_weights))

        # Reduce over ensemble dimension
        mse = mse.mean(1)

        #reduce over batch dimension
        mse = mse.mean(0)

                # Squash - reduce spatial and feature dimensions
        if squash:
            mse = self.sum_function(mse, axis=squash if isinstance(squash, tuple) else (-3, -2, -1))

        return mse

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss for logging."""
        return "mse"
