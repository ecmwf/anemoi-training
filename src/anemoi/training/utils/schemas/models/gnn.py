# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import field_validator

from anemoi.training.utils.schemas.utils import HydraInstantiable

from .base_model import BaseModelConfig


class GNNModelComponent(HydraInstantiable):
    activation: str = "GELU"
    trainable_size: NonNegativeInt = 8
    sub_graph_edge_attributes: list = Field(default_factory=list)
    mlp_extra_layers: int = 0
    num_chunks: NonNegativeInt = 1


class GNNProcessor(GNNModelComponent):
    num_layers: NonNegativeInt = 16
    num_chunks: NonNegativeInt = 2

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.models.layers.processor.GNNProcessor"
        return target


class GNNEncoder(GNNModelComponent):

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.models.layers.mapper.GNNForwardMapper"
        return target


class GNNDecoder(GNNModelComponent):

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.models.layers.mapper.GNNBackwardMapper"
        return target


class GNNConfig(BaseModelConfig):
    processor: GNNProcessor = Field(default_factory=GNNProcessor)
    encoder: GNNEncoder = Field(default_factory=GNNEncoder)
    decoder: GNNDecoder = Field(default_factory=GNNDecoder)
