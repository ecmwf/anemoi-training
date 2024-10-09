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
from dataclasses import field


@dataclass
class Eval:
    enabled: bool
    rollout: int
    frequency: int


@dataclass
class Plot:
    enabled: bool
    asynchronous: bool
    frequency: int
    sample_idx: int
    per_sample: int
    parameters: list[str]
    accumulation_levels_plot: list[int | float]
    cmap_accumulation: list[str]
    precip_and_related_fields: list[str]
    parameters_histogram: list[str]
    parameters_spectrum: list[str]
    parameter_groups: dict[str, list[str]]
    learned_features: bool


@dataclass
class Debug:
    anomaly_detection: bool


@dataclass
class Checkpoint:
    save_frequency: int | None
    num_models_saved: int


@dataclass
class Wandb:
    enabled: bool
    offline: bool
    log_model: bool
    project: str
    entity: str | None
    gradients: bool
    parameters: bool


@dataclass
class Mlflow:
    enabled: bool
    offline: bool
    authentication: bool
    log_model: bool
    tracking_uri: str | None
    experiment_name: str
    project_name: str
    system: bool
    terminal: bool
    run_name: str | None
    on_resume_create_child: bool


@dataclass
class Logging:
    wandb: Wandb
    tensorboard: dict[str, bool]
    mlflow: Mlflow
    interval: int


@dataclass
class DiagnosticsConfig:
    eval: Eval
    plot: Plot
    debug: Debug
    profiler: bool
    log: Logging
    enable_progress_bar: bool
    print_memory_summary: bool
    checkpoint: dict[str, Checkpoint] = field(default_factory=dict)
