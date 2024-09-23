# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class GradientClip:
    """Gradient clipping configuration."""

    val: float = 32.0
    algorithm: str = "value"


@dataclass
class SWA:
    enabled: bool = False
    lr: float = 1.0e-4


@dataclass
class Rollout:
    start: int = 1
    epoch_increment: int = 0
    max: int = 1


@dataclass
class LR:
    rate: float = 0.625e-4
    iterations: int = 300000
    min: float = 3e-7


@dataclass
class PressureLevel:
    q: float = 0.6
    t: float = 6
    u: float = 0.8
    v: float = 0.5
    w: float = 0.001
    z: float = 12


@dataclass
class Surface:
    sp: float = 10
    u10: float = 0.1
    v10: float = 0.1
    d2: float = 0.5
    tp: float = 0.025
    cp: float = 0.0025


@dataclass
class LossScaling:
    default: int = 1
    pl: PressureLevel = field(default_factory=PressureLevel)
    sfc: Surface = field(default_factory=Surface)


@dataclass
class PressureLevelScaler:
    _target_: str = "anemoi.training.data.scaling.ReluPressureLevelScaler"
    minimum: float = 0.2
    slope: float = 0.001


@dataclass
class TrainingConfig:
    """Training configuration."""

    run_id: str | None = None
    fork_run_id: str | None = None
    load_weights_only: bool = False
    deterministic: bool = False
    precision: str = "16-mixed"
    multistep_input: int = 4
    accum_grad_batches: int = 1
    gradient_clip: GradientClip = field(default_factory=GradientClip)
    swa: SWA = field(default_factory=SWA)
    zero_optimizer: bool = False
    loss_gradient_scaling: bool = False
    rollout: Rollout = field(default_factory=Rollout)
    max_epochs: int = 200
    lr: LR = field(default_factory=LR)
    loss_scaling: LossScaling = field(default_factory=LossScaling)
    metrics: list[str] = field(default_factory=list)
    pressure_level_scaler: PressureLevelScaler = field(default_factory=PressureLevelScaler)
