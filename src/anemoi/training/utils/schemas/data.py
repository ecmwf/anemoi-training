# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class NormalizerConfig:
    default: str
    min_max: str | None = None
    max: list[str] = field(default_factory=list)
    none: list[str] = field(default_factory=list)


@dataclass
class Processor:
    _target_: str
    _convert_: str
    config: str


@dataclass
class Normalizer(Processor):
    _target_: str
    _convert_: str
    config: NormalizerConfig


@dataclass
class Imputer(Processor):
    _target_: str
    _convert_: str
    config: dict[str, str]


@dataclass
class Remapper(Processor):
    _target_: str
    _convert_: str
    config: dict[str, str]


@dataclass
class DataConfig:
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
    remapped : Optional[str]
        The remapped features, if any.
    processors : Dict[str, Processor]
        The Processors configuration.
    num_features : Optional[int]
        The number of features in the forecast state. To be set in the code.
    """

    format: str
    resolution: str
    frequency: str
    timestep: str
    processors: dict
    forcing: list[str] = field(default_factory=list)
    diagnostic: list[str] = field(default_factory=list)
    remapped: str | None = None
    num_features: int | None = None
