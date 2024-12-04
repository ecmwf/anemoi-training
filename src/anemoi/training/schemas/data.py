# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import model_validator


class NormalizerSchema(BaseModel):
    default: str = Field(
        description="Normalizer default method to apply",
        examples=["mean-std", "std", "min-max", "max", "none"],
    )
    min_max: str | None = Field(None, description="Variables to normalize with min-max method.")
    max: list[str] = Field(default_factory=list, description="Variables to normalize with max method.")
    none: list[str] = Field(default_factory=list, description="Variables not to be normalized.")


class ImputerSchema(BaseModel):
    default: str = Field(description="Imputer default method to apply.", examples=["none", "mean", "stdev"])


class RemapperSchema(BaseModel):
    default: str = Field(description="Remapper default method to apply.", examples=["none", "cos_sin"])


class Target(str, Enum):
    normalizer = "anemoi.models.preprocessing.normalizer.InputNormalizer"
    imputer = "anemoi.models.preprocessing.imputer.InputImputer"
    remapper = "anemoi.models.preprocessing.remapper.Remapper"


target_to_schema = {Target.normalizer: NormalizerSchema, Target.imputer: ImputerSchema, Target.remapper: RemapperSchema}


class Processor(BaseModel):
    target_: Target = Field(
        ...,
        alias="_target_",
        description="Processor object from anemoi.models.preprocessing.[normalizer|imputer|remapper].",
    )
    config: NormalizerSchema | ImputerSchema | RemapperSchema = Field(
        description="Target schema containing processor methods.",
    )

    @model_validator(mode="after")
    def schema_consistent_with_target(self) -> Processor:
        if self.target_ not in target_to_schema or target_to_schema[self.target_] != self.config.__class__:
            error_msg = f"Schema {self.config.__class__} does not match target {self.target_}"
            raise ValidationError(error_msg)
        return self


class DataSchema(BaseModel):
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

    format: str = Field(None, description="Format of the data.")
    resolution: str = Field(None, description="Grid type and resolution.", examples=["o96", "n320", "o48"])
    frequency: str = Field(None, description="Time frequency requested from the dataset.")
    timestep: str = Field(None, description="Time step of model (must be multiple of frequency).")
    processors: dict[str, Processor] = Field(
        description="Layers of model performing computation on latent space. \
                                Processors including imputers and normalizers are applied in order of definition.",
    )
    forcing: list[str] = Field(
        default_factory=list,
        description="Features that are not part of the forecast state but are used as forcing to generate the \
                     forecast state.",
    )
    diagnostic: list[str] = Field(
        default_factory=list,
        description="Features that are only part of the forecast state and are not used as an input to the model.",
    )
    num_features: int | None = Field(
        None,
        description="Number of features in the forecast state.",
    )  # Set in the code should not be in the config???
    remapped: dict | None = Field(None, description="Dictionary of remapped names for variables.")
