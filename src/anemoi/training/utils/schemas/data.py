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

from .base_config import HydraInstantiable


class NormalizerConfig(BaseModel):
    default: str
    min_max: str | None = None
    max: list[str] = Field(default_factory=list)
    none: list[str] = Field(default_factory=list)


class ImputerConfig(BaseModel):
    default: str


class RemapperConfig(BaseModel):
    default: str


class Processor(HydraInstantiable):
    config: NormalizerConfig | ImputerConfig | RemapperConfig


class DataConfig(BaseModel):
    """A class used to represent the overall configuration of the dataset.

    Attributes
    ----------
    format : str
        The format of the data.
    resolution : str
        The resolution of the data.
    frequency : str
        The frequency of the data.
    timestep : str
        The timestep of the data.
    forcing : List[str]
        The list of features used as forcing to generate the forecast state.
    diagnostic : List[str]
        The list of features that are only part of the forecast state.
    processors : Dict[str, Processor]
        The Processors configuration.
    num_features : Optional[int]
        The number of features in the forecast state. To be set in the code.
    """

    format: str
    resolution: str
    frequency: str
    timestep: str
    processors: dict[str, Processor]
    forcing: list[str] = Field(default_factory=list)
    diagnostic: list[str] = Field(default_factory=list)
    num_features: int | None = None
    remapped: dict | None = None
