# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt


class AllowedModels(str, Enum):
    ANEMOI_MODEL_ENC_PROC_DEC = "anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"


class Model(BaseModel):
    target_: AllowedModels = Field(..., alias="_target_")
    convert_: str = Field("all", alias="_convert_")


class TransformerModelComponent(BaseModel):
    activation: str = Field(default="GELU")
    trainable_size: NonNegativeInt = Field(default=8)
    num_chunks: NonNegativeInt = Field(default=1)
    mlp_hidden_ratio: NonNegativeInt = Field(default=4)
    num_heads: NonNegativeInt = Field(default=16)  # GraphTransformer or Transformer only


class GraphTransformerEncoder(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GraphTransformerForwardMapper"] = Field(..., alias="_target_")
    sub_graph_edge_attributes: list[str] = ["edge_length", "edge_dir"]


class GraphTransformerDecoder(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GraphTransformerBackwardMapper"] = Field(..., alias="_target_")
    sub_graph_edge_attributes: list[str] = ["edge_length", "edge_dir"]


class TrainableParameters(BaseModel):
    data: NonNegativeInt = Field(default=8)
    hidden: NonNegativeInt = Field(default=8)


class BaseModelConfig(BaseModel):
    num_channels: NonNegativeInt = Field(default=512)
    model: Model = Field(default_factory=Model)
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    node_loss_weight: str = Field(default="area_weight")
