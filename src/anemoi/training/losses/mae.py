# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging

import torch

from anemoi.training.losses.weightedloss import FunctionalWeightedLoss

LOGGER = logging.getLogger(__name__)


class WeightedMAELoss(FunctionalWeightedLoss):
    """Node-weighted MAE loss."""

    name = "wmae"

    def calculate_difference(self, pred, target):
        return torch.abs(pred - target)
