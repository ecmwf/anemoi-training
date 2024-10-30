# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

from pydantic import BaseModel
from pydantic import NonNegativeInt
from pydantic import field_validator
from pydantic import model_validator


class Checkpoint(BaseModel):
    every_n_epochs: str
    every_n_train_steps: str
    every_n_minutes: str


class FilesConfig(BaseModel):
    dataset: Path
    graph: Path
    checkpoint: dict[str, str]
    warm_start: str | None


class Logs(BaseModel):
    base: str | None = None
    wandb: str | None = None
    mlflow: str | None = None
    tensorboard: str | None = None


class PathsConfig(BaseModel):
    data: str
    grids: str
    output: str
    logs: Logs
    checkpoints: str
    plots: str
    profiler: str
    graph: str


class HardwareConfig(BaseModel):
    accelerator: str
    num_gpus_per_node: NonNegativeInt
    num_nodes: NonNegativeInt
    num_gpus_per_model: NonNegativeInt
    files: FilesConfig
    paths: PathsConfig

    @field_validator("num_gpus_per_node")
    @classmethod
    def check_valid_num_gpus_per_node(cls, num_gpus_per_node: int) -> int:
        assert num_gpus_per_node <= 8, "num_gpus_per_node must be less than 8"
        return num_gpus_per_node

    @model_validator(mode="before")
    @classmethod
    def check_valid_num_gpus_per_model(cls, data: dict) -> dict:
        assert (
            data["num_gpus_per_model"] <= data["num_gpus_per_node"]
        ), "num_gpus_per_model must be less than num_gpus_per_node"
        return data
