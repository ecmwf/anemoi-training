# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt


class GradientClip(BaseModel):
    """Gradient clipping configuration."""

    val: float = 32.0
    algorithm: str = "value"


class SWA(BaseModel):
    enabled: bool = False
    lr: NonNegativeFloat = 1.0e-4


class Rollout(BaseModel):
    start: PositiveInt = 1
    epoch_increment: NonNegativeInt = 0
    max: PositiveInt = 1


class LR(BaseModel):
    rate: NonNegativeFloat = 0.625e-4
    iterations: NonNegativeInt = 300000
    min: NonNegativeFloat = 3e-7


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
    target_: str = Field("anemoi.training.data.scaling.ReluPressureLevelScaler", alias="_target_")
    minimum: float = 0.2
    slope: float = 0.001


class TrainingLossSchema(BaseModel):
    target_: str = Field("anemoi.training.losses.mse.WeightedMSELoss", alias="_target_")
    scalers: Any | None = None  # list[str]
    ignore_nans: bool = False


class ValidationMetricsSchema(BaseModel):
    target_: str = Field("anemoi.training.losses.mse.WeightedMSELoss", alias="_target_")
    scalers: Any | None = None  # list[str] | None = None
    ignore_nans: bool = True


class TrainingSchema(BaseModel):
    """Training configuration."""

    run_id: str | None = None
    fork_run_id: str | None = None
    load_weights_only: bool = False
    deterministic: bool = False
    precision: str = "16-mixed"
    multistep_input: PositiveInt = 4
    accum_grad_batches: PositiveInt = 1
    num_sanity_val_steps: PositiveInt = 6
    gradient_clip: GradientClip = Field(default_factory=GradientClip)
    swa: SWA = Field(default_factory=SWA)
    zero_optimizer: bool = False
    training_loss: TrainingLossSchema
    loss_gradient_scaling: bool = False
    validation_metrics: list[ValidationMetricsSchema] = Field(default_factory=ValidationMetricsSchema)
    rollout: Rollout = Field(default_factory=Rollout)
    max_epochs: PositiveInt | None = None
    max_steps: PositiveInt = 150000
    lr: LR = Field(default_factory=LR)
    loss_scaling: LossScaling = Field(default_factory=LossScaling)
    pressure_level_scaler: PressureLevelScaler = Field(default_factory=PressureLevelScaler)
