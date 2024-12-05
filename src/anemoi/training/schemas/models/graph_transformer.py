# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Literal

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt

from .base_model import BaseModelConfig
from .base_model import GraphTransformerDecoder
from .base_model import GraphTransformerEncoder
from .base_model import TransformerModelComponent


class GraphTransformerProcessor(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.GraphTransformerProcessor"] = Field(..., alias="_target_")
    sub_graph_edge_attributes: tuple[str] = ["edge_length", "edge_dir"]
    num_layers: NonNegativeInt = Field(default=16)
    num_chunks: NonNegativeInt = Field(default=2)
    dropout_p: NonNegativeFloat = Field(default=0.0)


class GraphTransformerConfig(BaseModelConfig):
    processor: GraphTransformerProcessor = Field(default_factory=GraphTransformerProcessor)
    encoder: GraphTransformerEncoder = Field(default_factory=GraphTransformerEncoder)
    decoder: GraphTransformerDecoder = Field(default_factory=GraphTransformerDecoder)
