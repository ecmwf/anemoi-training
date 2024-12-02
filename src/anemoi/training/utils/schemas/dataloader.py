# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import datetime  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any

from anemoi.utils.dates import frequency_to_timedelta
from pydantic import BaseModel
from pydantic import Field
from pydantic import PositiveInt
from pydantic import RootModel
from pydantic import computed_field


class Frequency(RootModel):
    root: Any

    @computed_field
    def as_timedelta(self) -> datetime.timedelta:
        return frequency_to_timedelta(self.root)

    @computed_field
    def as_string(self) -> str:

        total_seconds = self.as_seconds
        assert int(total_seconds) == total_seconds, total_seconds
        total_seconds = int(total_seconds)

        seconds = total_seconds

        days = seconds // (24 * 3600)
        seconds %= 24 * 3600
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        if days > 0 and hours == 0 and minutes == 0 and seconds == 0:
            return f"{days}d"

        if days == 0 and hours > 0 and minutes == 0 and seconds == 0:
            return f"{hours}h"

        if days == 0 and hours == 0 and minutes > 0 and seconds == 0:
            return f"{minutes}m"

        if days == 0 and hours == 0 and minutes == 0 and seconds > 0:
            return f"{seconds}s"

        if days > 0:
            return f"{total_seconds}s"

        return str(self.as_timedelta)

    @computed_field
    def as_seconds(self) -> int:
        return int(self.as_timedelta.total_seconds())


class DatasetSchema(BaseModel):
    """Dataset configuration schema."""

    dataset: dict | Path = Field(description="Path to dataset.")
    start: int | None = Field(None, description="Starting year for sample of the dataset.")
    end: int | None = Field(None, description="Ending year [inclusive] for sample of the dataset.")
    frequency: Frequency = Field(description="Temporal resolution, frequency must be >= to dataset frequency.")
    drop: list | None = Field(None, description="???")


class LoaderSet(BaseModel):
    training: PositiveInt | None = Field(None, description="Value for training dataset")
    validation: PositiveInt | None = Field(None, description="Value for validation dataset")
    test: PositiveInt | None = Field(None, description="Value for test dataset")


class DataLoaderSchema(BaseModel):
    prefetch_factor: int = Field(2, description="Number of batches loaded in advance by each worker.", ge=0)
    pin_memory: bool = Field(
        True,
        description="If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.",
    )

    num_workers: LoaderSet = Field(description="Number of process per-GPU for batch distribution.")
    batch_size: LoaderSet = Field(description="Per-GPU batch size.")
    limit_batches: LoaderSet = Field(
        None,
        description="Limit number of batches to run. Default value null, will run on all the batches.",
    )

    training: DatasetSchema = Field(None, description="Training DatasetSchema.")
    validation: DatasetSchema = Field(None, description="Validation DatasetSchema.")
    test: DatasetSchema = Field(None, description="Test DatasetSchema.")
    validation_rollout: PositiveInt = Field(
        1,
        description="Number of rollouts to use for validation, must be equal or greater than rollout expected \
                     by callbacks.",
    )  # TODO(Helen): Ccheck that this equal or greater than the number of rollouts expected by callbacks ???
