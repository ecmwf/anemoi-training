# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging as pl_StochasticWeightAveraging

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class StochasticWeightAveraging(pl_StochasticWeightAveraging):
    """Provide StochasticWeightAveraging from pytorch_lightning as a callback using config."""

    def __init__(self, config: DictConfig):
        super().__init__(
            swa_lrs=config.training.swa.lr,
            swa_epoch_start=min(
                int(0.75 * config.training.max_epochs),
                config.training.max_epochs - 1,
            ),
            annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
            annealing_strategy="cos",
            device=None,
        )
        self.config = config
