# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class GradientClip(BaseModel):
    """Gradient clipping configuration."""

    val: float = 32.0
    algorithm: str = "value"


class SWA(BaseModel):
    enabled: bool = False
    lr: float = 1.0e-4


class Rollout(BaseModel):
    start: int = 1
    epoch_increment: int = 0
    max: int = 1


class LR(BaseModel):
    rate: float = 0.625e-4
    iterations: int = 300000
    min: float = 3e-7


class PressureLevel(BaseModel):
    q: float | None = 0.6
    t: float | None = 6
    u: float | None = 0.8
    v: float | None = 0.5
    w: float | None = 0.001
    z: float | None = 12


class Surface(BaseModel):
    sp: float | None = 10
    u10: float | None = 0.1
    v10: float | None = 0.1
    d2: float | None = 0.5
    tp: float | None = 0.025
    cp: float | None = 0.0025


class LossScaling(BaseModel):
    default: int = 1
    pl: PressureLevel = Field(default_factory=PressureLevel)
    sfc: Surface = Field(default_factory=Surface)


class PressureLevelScaler(BaseModel):
    _target_: str = "anemoi.training.data.scaling.ReluPressureLevelScaler"
    minimum: float = 0.2
    slope: float = 0.001


class TrainingConfig(BaseModel):
    """Training configuration."""

    run_id: str | None = None
    fork_run_id: str | None = None
    load_weights_only: bool = False
    deterministic: bool = False
    precision: str = "16-mixed"
    multistep_input: int = 4
    accum_grad_batches: int = 1
    gradient_clip: GradientClip = Field(default_factory=GradientClip)
    swa: SWA = Field(default_factory=SWA)
    zero_optimizer: bool = False
    loss_gradient_scaling: bool = False
    rollout: Rollout = Field(default_factory=Rollout)
    max_epochs: int = 200
    lr: LR = Field(default_factory=LR)
    loss_scaling: LossScaling = Field(default_factory=LossScaling)
    metrics: list[str] = Field(default_factory=list)
    pressure_level_scaler: PressureLevelScaler = Field(default_factory=PressureLevelScaler)
