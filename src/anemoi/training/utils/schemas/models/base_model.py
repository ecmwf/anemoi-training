# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from dataclasses import dataclass
from dataclasses import field


@dataclass
class Model:
    _target_: str = "anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"


@dataclass
class TrainableParameters:
    data: int = 8
    hidden: int = 8
    data2hidden: int = 8
    hidden2data: int = 8


@dataclass
class Attributes:
    edges: list[str] = field(default_factory=lambda: ["edge_length", "edge_dirs"])
    nodes: list[str] = field(default_factory=list)


@dataclass
class BaseModelConfig:
    activation: str = "GELU"
    num_channels: int = 512
    model: Model = field(default_factory=Model)
    trainable_parameters: TrainableParameters = field(default_factory=TrainableParameters)
    attributes: Attributes = field(default_factory=Attributes)
    node_loss_weight: str = "area_weight"
