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

from .base_model import BaseModelConfig
from .base_model import GraphTransformerDecoder
from .base_model import GraphTransformerEncoder
from .base_model import TransformerModelComponent


class GraphTransformerProcessor(TransformerModelComponent):
    target_: str = Field("anemoi.models.layers.processor.GraphTransformerProcessor", alias="_target_")
    sub_graph_edge_attributes: list = Field(default_factory=list)
    num_layers: NonNegativeInt = 16
    num_chunks: NonNegativeInt = 2
    dropout_p: float = 0.0

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.models.layers.processor.GraphTransformerProcessor"
        return target


class GraphTransformerConfig(BaseModelConfig):
    processor: GraphTransformerProcessor = Field(default_factory=GraphTransformerProcessor)
    encoder: GraphTransformerEncoder = Field(default_factory=GraphTransformerEncoder)
    decoder: GraphTransformerDecoder = Field(default_factory=GraphTransformerDecoder)
