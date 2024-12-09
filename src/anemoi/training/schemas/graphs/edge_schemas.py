# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import field_validator


class KNNEdgeSchema(BaseModel):
    target_: str = Field("anemoi.graphs.edges.KNNEdges", alias="_target_")
    num_nearest_neighbours: PositiveInt = Field(default=3)

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.graphs.edges.KNNEdges"
        return target


class CutoffEdgeSchema(BaseModel):
    target_: str = Field("anemoi.graphs.edges.CutOffEdges", alias="_target_")
    cutoff_factor: PositiveFloat = Field(default=0.6)

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.graphs.edges.CutOffEdges"
        return target


class MultiScaleEdgeSchema(BaseModel):
    target_: str = Field("anemoi.graphs.edges.MultiScaleEdges", alias="_target_")
    x_hops: PositiveInt = Field(default=1)

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.graphs.edges.MultiScaleEdges"
        return target


class EdgeAttributeSchema(BaseModel):
    target_: str = Field("anemoi.graphs.edges.attributes.EdgeLength", alias="_target_")
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(default="unit-std")

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target in ["anemoi.graphs.edges.attributes.EdgeLength", "anemoi.graphs.edges.attributes.EdgeDirection"]
        return target
