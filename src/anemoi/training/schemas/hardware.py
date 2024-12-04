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
from pydantic import DirectoryPath
from pydantic import NonNegativeInt
from pydantic import field_validator
from pydantic import model_validator


class Checkpoint(BaseModel):
    every_n_epochs: str = "anemoi-by_epoch-epoch_{epoch:03d}-step_{step:06d}"
    every_n_train_steps: str = "anemoi-by_step-epoch_{epoch:03d}-step_{step:06d}"
    every_n_minutes: str = "anemoi-by_time-epoch_{epoch:03d}-step_{step:06d}"


class FilesConfig(BaseModel):
    dataset: Path  # TODO(Helen): Change to FilePath, only posisble after refactor
    graph: Path
    checkpoint: dict[str, str]
    warm_start: str | None = None


class Logs(BaseModel):
    base: DirectoryPath | None = None
    wandb: DirectoryPath | None = None
    mlflow: DirectoryPath | None = None
    tensorboard: DirectoryPath | None = None


class PathsConfig(BaseModel):
    data: DirectoryPath
    grids: DirectoryPath
    graph: DirectoryPath
    output: Path
    logs: Logs
    checkpoints: Path
    plots: Path
    profiler: Path


class HardwareSchema(BaseModel):
    accelerator: str = "auto"
    num_gpus_per_node: NonNegativeInt = 1
    num_nodes: NonNegativeInt = 1
    num_gpus_per_model: NonNegativeInt = 1
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
