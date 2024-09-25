# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from dataclasses import dataclass
from typing import Any

from .data import DataConfig
from .hardware import HardwareConfig
from .models.gnn import GNNConfig
from .training import TrainingConfig


@dataclass
class BaseConfig:
    data: DataConfig
    dataloader: Any
    diagnostics: Any
    hardware: HardwareConfig
    graph: Any
    model: GNNConfig
    training: TrainingConfig
