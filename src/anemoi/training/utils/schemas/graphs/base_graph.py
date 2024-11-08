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
from pydantic import field_validator

from .edge_schemas import CutoffEdgeSchema  # noqa: TCH001
from .edge_schemas import EdgeAttributeSchema  # noqa: TCH001
from .edge_schemas import KNNEdgeSchema  # noqa: TCH001
from .edge_schemas import MultiScaleEdgeSchema  # noqa: TCH001
from .node_schemas import IcosahedralNodeSchema  # noqa: TCH001
from .node_schemas import LimitedAreaNPZFileNodesSchema  # noqa: TCH001
from .node_schemas import NPZnodeSchema  # noqa: TCH001
from .node_schemas import ZarrNodeSchema  # noqa: TCH001


class AreaWeightSchema(BaseModel):
    target_: str = Field("anemoi.graphs.nodes.attributes.AreaWeights", alias="_target_")
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(default="unit-max")

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.graphs.nodes.attributes.AreaWeights"
        return target


class NodeSchema(BaseModel):
    node_builder: ZarrNodeSchema | NPZnodeSchema | IcosahedralNodeSchema | LimitedAreaNPZFileNodesSchema
    attributes: dict[str, AreaWeightSchema] | None = None


class EdgeSchema(BaseModel):
    source_name: str
    target_name: str
    edge_builder: CutoffEdgeSchema | KNNEdgeSchema | MultiScaleEdgeSchema
    attributes: dict[str, EdgeAttributeSchema]


class BaseGraphConfig(BaseModel):

    nodes: dict[str, NodeSchema]
    edges: list[EdgeSchema]
    overwrite: bool = True
    data: str = "data"
    hidden: str = "hidden"