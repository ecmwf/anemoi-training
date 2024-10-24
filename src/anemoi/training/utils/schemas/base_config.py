# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from typing import Any

from pydantic import BaseModel

from .data import DataConfig
from .diagnostics import DiagnosticsConfig
from .hardware import HardwareConfig
from .training import TrainingConfig


class BaseConfig(BaseModel):
    data: DataConfig
    dataloader: Any
    diagnostics: DiagnosticsConfig
    hardware: HardwareConfig
    graph: Any  # BaseGraphConfig
    model: Any
    training: TrainingConfig
