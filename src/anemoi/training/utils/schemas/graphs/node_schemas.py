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
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import field_validator


class ZarrNodeSchema(BaseModel):
    target_: str = Field("anemoi.graphs.nodes.ZarrDatasetNodes", alias="_target_")
    dataset: str | dict  # TODO(Helen): Discuss schema with Baudouin

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.graphs.nodes.ZarrDatasetNodes"
        return target


class NPZnodeSchema(BaseModel):
    target_: str = Field("anemoi.graphs.nodes.NPZFileNodes", alias="_target_")
    grid_definition_path: str | None = None
    resolution: str | None = None

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.graphs.nodes.NPZFileNodes"
        return target


class LimitedAreaNPZFileNodesSchema(BaseModel):
    target_: str = Field("anemoi.graphs.nodes.LimitedAreaNPZFileNodes", alias="_target_")
    grid_definition_path: str
    resolution: str
    reference_node_name: str  # TODO(Helen): Discuss check that reference nodes exists in the config
    name: str
    mask_attr_name: str  # TODO(Helen): Discuss check that mask_attr_name exists in the dataset config
    margin_radius_km: PositiveFloat = Field(default=100.0)

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert target == "anemoi.graphs.nodes.LimitedAreaNPZFileNodes"
        return target


class IcosahedralandHealPixNodeSchema(BaseModel):
    target_: str = Field("anemoi.graphs.nodes.TriNodes", alias="_target_")
    resolution: PositiveInt | None = None

    @field_validator("target_")
    @classmethod
    def check_valid_target(cls, target: str) -> str:
        assert (
            target == "anemoi.graphs.nodes.TriNodes"
            or target == "anemoi.graphs.nodes.HexNodes"
            or target == "anemoi.graphs.nodes.HEALPixNodes"
        )
        return target
