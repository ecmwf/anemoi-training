# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt

from .base_model import BaseModelConfig


class GNNModelComponent(BaseModel):
    activation: str = Field(default="GELU")
    trainable_size: NonNegativeInt = Field(default=8)
    num_chunks: NonNegativeInt = Field(default=1)
    sub_graph_edge_attributes: list = Field(default_factory=list)
    mlp_extra_layers: NonNegativeInt = Field(default=0)


class GNNProcessor(GNNModelComponent):
    target_: Literal["anemoi.models.layers.processor.GNNProcessor"] = Field(..., alias="_target_")
    num_layers: NonNegativeInt = Field(default=16)
    num_chunks: NonNegativeInt = Field(default=2)


class GNNEncoder(GNNModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GNNForwardMapper"] = Field(..., alias="_target_")


class GNNDecoder(GNNModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GNNBackwardMapper"] = Field(..., alias="_target_")


class GNNConfig(BaseModelConfig):
    processor: GNNProcessor = Field(default_factory=GNNProcessor)
    encoder: GNNEncoder = Field(default_factory=GNNEncoder)
    decoder: GNNDecoder = Field(default_factory=GNNDecoder)
