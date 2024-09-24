# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pytorch_lightning.callbacks import LearningRateMonitor as pl_LearningRateMonitor

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class LearningRateMonitor(pl_LearningRateMonitor):
    """Provide LearningRateMonitor from pytorch_lightning as a callback."""

    def __init__(self, config: DictConfig):
        super().__init__(logging_interval="step", log_momentum=False)
        self.config = config
