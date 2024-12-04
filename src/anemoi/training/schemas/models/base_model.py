# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import field_validator

_allowed_models = ["anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"]


class HydraInstantiable(BaseModel):
    target_: str = Field(..., alias="_target_")
    convert_: str = Field("all", alias="_convert_")


class TransformerModelComponent(BaseModel):
    activation: str = Field(default="GELU")
    trainable_size: NonNegativeInt = Field(default=8)
    num_chunks: NonNegativeInt = Field(default=1)
    mlp_hidden_ratio: NonNegativeInt = Field(default=4)
    num_heads: NonNegativeInt = Field(default=16)  # GraphTransformer or Transformer only


class GraphTransformerEncoder(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.mapper.GraphTransformerForwardMapper", alias="_target_")
    sub_graph_edge_attributes: list[str] = ["edge_length", "edge_dir"]

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.models.layers.mapper.GraphTransformerForwardMapper"
        return target


class GraphTransformerDecoder(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.mapper.GraphTransformerBackwardMapper", alias="_target_")
    sub_graph_edge_attributes: list[str] = ["edge_length", "edge_dir"]

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.models.layers.mapper.GraphTransformerBackwardMapper"
        return target


class TrainableParameters(BaseModel):
    data: NonNegativeInt = Field(default=8)
    hidden: NonNegativeInt = Field(default=8)


class BaseModelConfig(BaseModel):
    num_channels: NonNegativeInt = Field(default=512)
    model: HydraInstantiable = Field(default_factory=HydraInstantiable)
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    node_loss_weight: str = Field(default="area_weight")

    @field_validator("model")
    @classmethod
    def check_valid_model(cls, model: HydraInstantiable) -> HydraInstantiable:
        assert model.target_ in _allowed_models, f"Model not implemented. Allowed models are {_allowed_models}"
        return model
