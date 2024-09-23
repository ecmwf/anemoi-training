# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Checkpoint:
    every_n_epochs: str
    every_n_train_steps: str
    every_n_minutes: str


@dataclass
class FilesConfig:
    dataset: str
    graph: str
    checkpoint: Checkpoint
    warm_start: str | None


@dataclass
class Logs:
    base: str
    wandb: str
    mlflow: str
    tensorboard: str


@dataclass
class PathsConfig:
    data: str
    grids: str
    output: str
    logs: Logs
    checkpoints: str
    plots: str
    profiler: str
    graph: str


@dataclass
class HardwareConfig:
    accelerator: str
    num_gpus_per_node: int
    num_nodes: int
    num_gpus_per_model: int
    files: FilesConfig
    paths: PathsConfig
