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

from anemoi.training.utils.schemas.utils import HydraInstantiable

_allowed_models = ["anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"]


class TransformerModelComponent(BaseModel):
    activation: str = "GELU"
    trainable_size: NonNegativeInt = 8
    num_chunks: NonNegativeInt = 1
    mlp_hidden_ratio: NonNegativeInt = 4
    num_heads: NonNegativeInt = 16  # GraphTransformer or Transformer only


class GraphTransformerEncoder(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.mapper.GraphTransformerForwardMapper", alias="_target_")
    sub_graph_edge_attributes: list = Field(default_factory=list)


class GraphTransformerDecoder(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.mapper.GraphTransformerBackwardMapper", alias="_target_")
    sub_graph_edge_attributes: list = Field(default_factory=list)


class TrainableParameters(BaseModel):
    data: NonNegativeInt = 8
    hidden: NonNegativeInt = 8
    data2hidden: NonNegativeInt = 8
    hidden2data: NonNegativeInt = 8
    hidden2hidden: NonNegativeInt | None = None


class Attributes(BaseModel):
    edges: list[str] = Field(default_factory=lambda: ["edge_length", "edge_dirs"])
    nodes: list[str] = Field(default_factory=list)


class BaseModelConfig(BaseModel):
    activation: str = "GELU"
    num_channels: NonNegativeInt = 512
    model: HydraInstantiable = Field(default_factory=HydraInstantiable)
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    attributes: Attributes = Field(default_factory=Attributes)
    node_loss_weight: str = "area_weight"

    @field_validator("model")
    @classmethod
    def check_valid_model(cls, model: HydraInstantiable) -> HydraInstantiable:
        assert model.target_ in _allowed_models, f"Model not implemented. Allowed models are {_allowed_models}"
        return model
