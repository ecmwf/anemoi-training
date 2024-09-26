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

from .base_model import BaseModelConfig


@dataclass
class ModelComponent:
    _target_: str = ""
    _convert_: str = "all"
    activation: str = "${model.activation}"
    trainable_size: int = 8
    num_chunks: int = 1
    mlp_hidden_ratio: int = 4
    num_heads: int = 16  # GraphTransformer or Transformer only


@dataclass
class Processor(ModelComponent):
    _target_: str = "anemoi.models.layers.processor.TransformerProcessor "
    _convert_: str = "all"
    num_layers: int = 16
    num_chunks: int = 2
    window_size: int = 512
    dropout_p: float = 0.0  # GraphTransformer


@dataclass
class Encoder(ModelComponent):
    _target_: str = "anemoi.models.layers.mapper.GraphTransformerForwardMapper"
    trainable_size: int = "${model.trainable_parameters.data2hidden}"
    sub_graph_edge_attributes: list = field(default_factory=list)
    mlp_hidden_ratio: int = 4  # GraphTransformer or Transformer only


@dataclass
class Decoder(ModelComponent):
    _target_: str = "anemoi.models.layers.mapper.GraphTransformerBackwardMapper"
    trainable_size: int = "${model.trainable_parameters.hidden2data}"
    sub_graph_edge_attributes: list = field(default_factory=list)
    mlp_hidden_ratio: int = 4  # GraphTransformer or Transformer only


@dataclass
class TransformerConfig(BaseModelConfig):
    processor: Processor = field(default_factory=Processor)
    encoder: Encoder = field(default_factory=Encoder)
    decoder: Decoder = field(default_factory=Decoder)
