# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel

# to make these available at runtime for pydantic, bug should be resolved in
# future versions (see https://github.com/astral-sh/ruff/issues/7866)
from .data import DataConfig  # noqa: TCH001
from .diagnostics import DiagnosticsConfig  # noqa: TCH001
from .graphs.base_graph import BaseGraphConfig  # noqa: TCH001
from .hardware import HardwareConfig  # noqa: TCH001
from .models.gnn import GNNConfig  # noqa: TCH001
from .models.transformer import TransformerConfig  # noqa: TCH001
from .training import TrainingConfig  # noqa: TCH001


class BaseConfig(BaseModel):
    data: DataConfig
    dataloader: Any
    diagnostics: DiagnosticsConfig
    hardware: HardwareConfig
    graph: BaseGraphConfig
    model: GNNConfig | TransformerConfig
    training: TrainingConfig

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


def convert_to_omegaconf(config: BaseConfig) -> dict:

    config = {
        "data": config.data.model_dump(by_alias=True),
        "dataloader": config.dataloader,
        "diagnostics": config.diagnostics.model_dump(),
        "hardware": config.hardware.model_dump(),
        "graph": config.graph.model_dump(by_alias=True),
        "model": config.model.model_dump(by_alias=True),
        "training": config.training.model_dump(by_alias=True),
    }

    return OmegaConf.create(config)
