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
class ModelComponent:
    _target_: str = ""
    _convert_: str = "all"
    trainable_size: int = ""
    sub_graph_edge_attributes: list = field(default_factory=list)
    activation: str = "${model.activation}"
    num_chunks: int = 1
    mlp_hidden_ratio: int = 4  # GraphTransformer or Transformer only
    num_heads: int = 16  # GraphTransformer or Transformer only


@dataclass
class Processor(ModelComponent):
    _target_: str = "anemoi.models.layers.processor.GraphTransformerProcessor"
    _convert_: str = "all"
    trainable_size: int = "${model.trainable_parameters.hidden2hidden}"
    num_layers: int = 16
    num_chunks: int = 2
    dropout_p: float = 0.0


@dataclass
class Encoder(ModelComponent):
    _target_: str = "anemoi.models.layers.mapper.GraphTransformerForwardMapper"
    trainable_size: int = "${model.trainable_parameters.data2hidden}"


@dataclass
class Decoder(ModelComponent):
    _target_: str = "anemoi.models.layers.mapper.GraphTransformerBackwardMapper"
    trainable_size: int = "${model.trainable_parameters.hidden2data}"


@dataclass
class TrainableParameters:
    data: int = 8
    hidden: int = 8
    data2hidden: int = 8
    hidden2data: int = 8
    hidden2hidden: int = 8


@dataclass
class Attributes:
    edges: list[str] = field(default_factory=lambda: ["edge_length", "edge_dirs"])
    nodes: list[str] = field(default_factory=list)


@dataclass
class GraphTransformerConfig:
    activation: str = "GELU"
    num_channels: int = 512
    model: Model = field(default_factory=Model)
    processor: Processor = field(default_factory=Processor)
    encoder: Encoder = field(default_factory=Encoder)
    decoder: Decoder = field(default_factory=Decoder)
    trainable_parameters: TrainableParameters = field(default_factory=TrainableParameters)
    attributes: Attributes = field(default_factory=Attributes)
    node_loss_weight: str = "area_weight"
