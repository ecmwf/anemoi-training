# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt

from .base_model import BaseModelConfig


class TransformerModelComponent(BaseModel):
    activation: str = "GELU"
    trainable_size: NonNegativeInt = 8
    num_chunks: NonNegativeInt = 1
    mlp_hidden_ratio: NonNegativeInt = 4
    num_heads: NonNegativeInt = 16  # GraphTransformer or Transformer only


class Processor(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.processor.TransformerProcessor", alias="_target_")
    num_layers: NonNegativeInt = 16
    num_chunks: NonNegativeInt = 2
    window_size: NonNegativeInt = 512
    dropout_p: float = 0.0  # GraphTransformer


class Encoder(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.mapper.GraphTransformerForwardMapper", alias="_target_")
    sub_graph_edge_attributes: list = Field(default_factory=list)


class Decoder(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.mapper.GraphTransformerBackwardMapper", alias="_target_")
    sub_graph_edge_attributes: list = Field(default_factory=list)


class TransformerConfig(BaseModelConfig):
    processor: Processor = Field(default_factory=Processor)
    encoder: Encoder = Field(default_factory=Encoder)
    decoder: Decoder = Field(default_factory=Decoder)
