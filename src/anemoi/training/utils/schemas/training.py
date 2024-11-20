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
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt


class GradientClip(BaseModel):
    """Gradient clipping configuration."""

    val: float = 32.0
    algorithm: str = "value"


class SWA(BaseModel):
    """Stochastic weight averaging configuration."""

    enabled: bool = False
    lr: NonNegativeFloat = 1.0e-4


class Rollout(BaseModel):
    """Rollout configuration."""

    start: PositiveInt = 1
    epoch_increment: NonNegativeInt = 0
    max: PositiveInt = 1


class LR(BaseModel):
    """Learning rate configuration."""

    rate: NonNegativeFloat = 0.625e-4
    iterations: NonNegativeInt = 300000
    min: NonNegativeFloat = 3e-7


class LossScalingSchema(BaseModel):
    default: int = 1
    pl: dict[str, NonNegativeFloat]
    sfc: dict[str, NonNegativeFloat]


class PressureLevelScalerSchema(BaseModel):
    target_: str = Field("anemoi.training.data.scaling.ReluPressureLevelScaler", alias="_target_")
    minimum: float = 0.2
    slope: float = 0.001


class MetricLossSchema(BaseModel):
    target_: str = Field("anemoi.training.losses.mse.WeightedMSELoss", alias="_target_")
    scalars: list[str] = Field(default_factory=[])
    ignore_nans: bool = False


class TrainingSchema(BaseModel):
    """Training configuration."""

    run_id: str | None = None
    fork_run_id: str | None = None
    load_weights_only: bool = False
    deterministic: bool = False
    precision: str = "16-mixed"
    multistep_input: PositiveInt = 2
    accum_grad_batches: PositiveInt = 1
    num_sanity_val_steps: PositiveInt = 6
    gradient_clip: GradientClip = Field(default_factory=GradientClip)
    swa: SWA = Field(default_factory=SWA)
    zero_optimizer: bool = False
    training_loss: MetricLossSchema
    loss_gradient_scaling: bool = False
    validation_metrics: list[MetricLossSchema] = Field(default_factory=MetricLossSchema)
    rollout: Rollout = Field(default_factory=Rollout)
    max_epochs: PositiveInt | None = None
    max_steps: PositiveInt = 150000
    lr: LR = Field(default_factory=LR)
    loss_scaling: LossScalingSchema = Field(default_factory=LossScalingSchema)
    pressure_level_scaler: PressureLevelScalerSchema = Field(default_factory=PressureLevelScalerSchema)
    metrics: list[str]
