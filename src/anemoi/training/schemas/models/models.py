# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from enum import Enum
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import field_validator

from .decoder import GNNDecoder
from .decoder import GraphTransformerDecoder
from .encoder import GNNEncoder
from .encoder import GraphTransformerEncoder
from .processor import GNNProcessor
from .processor import GraphTransformerProcessor
from .processor import TransformerProcessor

LOGGER = logging.getLogger(__name__)


class AllowedModels(str, Enum):
    ANEMOI_MODEL_ENC_PROC_DEC = "anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"


class Model(BaseModel):
    target_: AllowedModels = Field(..., alias="_target_")
    "Model object defined in anemoi.models.model."
    convert_: str = Field("all", alias="_convert_")
    "Target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."


class TrainableParameters(BaseModel):
    data: NonNegativeInt = Field(default=8)
    "Size of the learnable data node tensor. Default to 8."
    hidden: NonNegativeInt = Field(default=8)
    "Size of the learnable hidden node tensor. Default to 8."


class ReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.ReluBounding"] = Field(..., alias="_target_")
    "Relu bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the Relu method."


class FractionBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.FractionBounding"] = Field(..., alias="_target_")
    "Fraction bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh fraction method."
    min_val: float
    "The minimum value for the HardTanh activation. Correspond to the minimum fraction of the total_var."
    max_val: float
    "The maximum value for the HardTanh activation. Correspond to the maximum fraction of the total_var."
    total_var: str
    "Variable from which the secondary variables are derived. \
    For example, convective precipitation should be a fraction of total precipitation."


class HardtanhBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.HardtanhBounding"] = Field(..., alias="_target_")
    "Hard tanh bounding method function from anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh method."
    min_val: float
    "The minimum value for the HardTanh activation."
    max_val: float
    "The maximum value for the HardTanh activation."


defined_boundings = [
    "anemoi.models.layers.bounding.HardtanhBounding",
    "anemoi.models.layers.bounding.FractionBounding",
    "anemoi.models.layers.bounding.ReluBounding",
]


class BaseModelConfig(BaseModel):
    num_channels: NonNegativeInt = Field(default=512)
    "Feature tensor size in the hidden space."
    model: Model = Field(default_factory=Model)
    "Model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    "Learnable node and edge parameters."
    bounding: list[ReluBoundingSchema | HardtanhBoundingSchema | FractionBoundingSchema | Any]
    "List of bounding configuration applied in order to the specified variables."

    @field_validator("bounding")
    @classmethod
    def validate_bounding_schema_exist(cls, boundings: list) -> list:
        for bounding in boundings:
            if bounding["_target_"] not in defined_boundings:
                LOGGER.warning("%s bounding schema is not defined in anemoi.", bounding["_target_"])
        return boundings


class GNNConfig(BaseModelConfig):
    processor: GNNProcessor = Field(default_factory=GNNProcessor)
    "GNN processor schema."
    encoder: GNNEncoder = Field(default_factory=GNNEncoder)
    "GNN encoder schema."
    decoder: GNNDecoder = Field(default_factory=GNNDecoder)
    "GNN decoder schema."


class GraphTransformerConfig(BaseModelConfig):
    processor: GraphTransformerProcessor = Field(default_factory=GraphTransformerProcessor)
    "Graph transformer processor schema."
    encoder: GraphTransformerEncoder = Field(default_factory=GraphTransformerEncoder)
    "Graph transformer encoder schema."
    decoder: GraphTransformerDecoder = Field(default_factory=GraphTransformerDecoder)
    "Graph transformer decoder schema."


class TransformerConfig(BaseModelConfig):
    processor: TransformerProcessor = Field(default_factory=TransformerProcessor)
    "Transformer processor schema."
    encoder: GraphTransformerEncoder = Field(default_factory=GraphTransformerEncoder)
    "Graph transformer encoder schema."
    decoder: GraphTransformerDecoder = Field(default_factory=GraphTransformerDecoder)
    "Graph transformer decoder schema."
