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

from pydantic import AnyUrl
from pydantic import BaseModel
from pydantic import Field
from pydantic import PositiveInt


class Plot(BaseModel):
    asynchronous: bool
    frequency: PlottingFrequency
    sample_idx: int
    parameters: list[str]
    precip_and_related_fields: list[str]
    callbacks: Any = Field(default=[])


class PlottingFrequency(BaseModel):
    batch: PositiveInt = Field(default=750)
    epoch: PositiveInt = Field(default=5)


class Debug(BaseModel):
    anomaly_detection: bool


class Checkpoint(BaseModel):
    save_frequency: int | None
    num_models_saved: int


class Wandb(BaseModel):
    enabled: bool
    offline: bool
    log_model: bool
    project: str
    gradients: bool
    parameters: bool
    entity: str | None = None


class Mlflow(BaseModel):
    enabled: bool
    offline: bool
    authentication: bool
    log_model: bool
    tracking_uri: AnyUrl | None
    experiment_name: str
    project_name: str
    system: bool
    terminal: bool
    run_name: str | None
    on_resume_create_child: bool
    expand_hyperparams: Any


class Tensorboard(BaseModel):
    enabled: bool


class Logging(BaseModel):
    wandb: Wandb
    tensorboard: Tensorboard
    mlflow: Mlflow
    interval: PositiveInt


class BenchmarkProfilerSchema(BaseModel):
    memory: Any
    time: Any
    speed: Any
    system: Any
    model_summary: Any
    snapshot: Any


class DiagnosticsSchema(BaseModel):
    plot: Plot | None = None
    callbacks: Any = Field(default=[])
    benchmark_profiler: BenchmarkProfilerSchema
    debug: Debug
    profiler: bool
    log: Logging
    enable_progress_bar: bool
    print_memory_summary: bool
    enable_checkpointing: bool
    checkpoint: dict[str, Checkpoint] = Field(default_factory=dict)
