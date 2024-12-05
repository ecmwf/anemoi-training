# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Annotated

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from anemoi.training.schemas.utils import allowed_values


class GradientClip(BaseModel):
    """Gradient clipping configuration."""

    val: float = 32.0
    "Gradient clipping value."
    algorithm: Annotated[str, AfterValidator(partial(allowed_values, values=["value", "norm"]))] = Field(
        default="value",
    )
    "The gradient clipping algorithm to use"


class SWA(BaseModel):
    """Stochastic weight averaging configuration.

    See https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
    """

    enabled: bool = Field(default=False)
    "Enable stochastic weight averaging."
    lr: NonNegativeFloat = Field(default=1.0e-4)
    "Learning rate for SWA."


class Rollout(BaseModel):
    """Rollout configuration."""

    start: PositiveInt = Field(default=1)
    "Number of rollouts to start with."
    epoch_increment: NonNegativeInt = Field(default=0)
    "Number of epochs to increment the rollout."
    max: PositiveInt = Field(default=1)
    "Maximum number of rollouts."


class LR(BaseModel):
    """Learning rate configuration.

    Changes in per-gpu batch_size should come with a rescaling of the local_lr,
    in order to keep a constant global_lr global_lr = local_lr * num_gpus_per_node * num_nodes / gpus_per_model.
    """

    rate: NonNegativeFloat = Field(default=0.625e-4)  # TODO(Helen): Could be computed by pydantic
    "Initial learning rate. Is adjusteed according to the hardware configuration"
    iterations: NonNegativeInt = Field(default=300000)
    "Number of iterations."
    min: NonNegativeFloat = Field(default=3e-7)
    "Minimum learning rate."
    warmup_time: NonNegativeInt = Field(default=1000)


class LossScalingSchema(BaseModel):
    default: int = 1
    pl: dict[str, NonNegativeFloat]
    sfc: dict[str, NonNegativeFloat]


class PressureLevelScalerTargets(str, Enum):

    relu_scaler = "anemoi.training.data.scaling.ReluPressureLevelScaler"
    linear_scaler = "anemoi.training.data.scaling.LinearPressureLevelScaler"
    polynomial_sclaer = "anemoi.training.data.scaling.PolynomialPressureLevelScaler"
    no_scaler = "anemoi.training.data.scaling.NoPressureLevelScaler"


class PressureLevelScalerSchema(BaseModel):
    target_: PressureLevelScalerTargets = Field(
        default="anemoi.training.data.scaling.ReluPressureLevelScaler",
        alias="_target_",
    )
    minimum: float = Field(default=0.2)
    "Minimum value of the scaling function."
    slope: float = 0.001
    "Slope of the scaling function."


class MetricLossSchema(BaseModel):
    target_: str = Field("anemoi.training.losses.mse.WeightedMSELoss", alias="_target_")
    scalars: list[str] = Field(default=["variable"])
    "Scalars to include in loss calculation"
    ignore_nans: bool = False
    "Allow nans in the loss and apply methods ignoring nans for measuring the loss."


class NodeLossWeightsTargets(str, Enum):
    graph_node_attribute = "anemoi.training.losses.nodeweights.GraphNodeAttribute"
    reweighted_graph_node_attributes = "anemoi.training.losses.ReweightedGraphNodeAttribute"


class NodeLossWeightsSchema(BaseModel):
    target_: NodeLossWeightsTargets = Field(..., alias="_target_")
    target_nodes: str
    node_attribute: str


class TrainingSchema(BaseModel):
    """Training configuration."""

    run_id: str | None = Field(default=None)
    "Run ID: used to resume a run from a checkpoint, either last.ckpt or specified in hardware.files.warm_start."
    fork_run_id: str | None = Field(default=None)
    "Run ID to fork from, either last.ckpt or specified in hardware.files.warm_start."
    load_weights_only: bool = Field(default=False)
    "Load only the weights from the checkpoint, not the optimiser state."
    deterministic: bool = Field(deafult=False)
    "This flag sets the torch.backends.cudnn.deterministic flag. Might be slower, but ensures reproducibility."
    precision: str = Field(deafult="16-mixed")
    "Precision"
    multistep_input: PositiveInt = Field(default=2)
    """Number of input steps for the model. E.g. 1 = single step scheme, X(t-1) used to predict X(t),
    k > 1: multistep scheme, uses [X(t-k), X(t-k+1), ... X(t-1)] to predict X(t)."""
    accum_grad_batches: PositiveInt = Field(deafult=1)
    """Accumulates gradients over k batches before stepping the optimizer.
    K >= 1 (if K == 1 then no accumulation). The effective bacthsize becomes num-device * k."""
    num_sanity_val_steps: PositiveInt = Field(default=6)
    "Sanity check runs n batches of val before starting the training routine."
    gradient_clip: GradientClip = Field(default_factory=GradientClip)
    "Config for gradient clipping."
    swa: SWA = Field(default_factory=SWA)
    "Config for stochastic weight averaging"
    zero_optimizer: bool = Field(default=False)
    "use ZeroRedundancyOptimizer, saves memory for larger models"
    training_loss: MetricLossSchema
    "Training loss configuration."
    loss_gradient_scaling: bool = False
    "Dynamic rescaling of the loss gradient. Not yet tested."
    validation_metrics: list[MetricLossSchema] = Field(default_factory=MetricLossSchema)
    "List of validation metrics configurations."
    rollout: Rollout = Field(default_factory=Rollout)
    "Rollout configuration."
    max_epochs: PositiveInt | None = None
    "Maximum number of epochs, stops earlier if max_steps is reached first."
    max_steps: PositiveInt = 150000
    "Maximum number of steps, stops earlier if max_epochs is reached first."
    lr: LR = Field(default_factory=LR)
    "Learning rate configuration."
    variable_loss_scaling: LossScalingSchema = Field(default_factory=LossScalingSchema)
    "Variable loss scaling configuration."
    pressure_level_scaler: PressureLevelScalerSchema = Field(default_factory=PressureLevelScalerSchema)
    "Pressure level scaler configuration."
    metrics: list[str]
    "List of metrics"
    node_loss_weights: NodeLossWeightsSchema
    "Node loss weights configuration."
