# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from functools import cached_property
from typing import Callable
from typing import Union

import pytorch_lightning as pl
from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.utils.dates import frequency_to_seconds
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.dataset import worker_init_func

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize Anemoi Datasets data module.

        Parameters
        ----------
        config : DictConfig
            Job configuration

        """
        super().__init__()

        self.config = config

        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))  # global rank
        self.model_comm_group_id = (
            self.global_rank // self.config.hardware.num_gpus_per_model
        )  # id of the model communication group the rank is participating in
        self.model_comm_group_rank = (
            self.global_rank % self.config.hardware.num_gpus_per_model
        )  # rank within one model communication group
        total_gpus = self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes
        assert (
            total_gpus
        ) % self.config.hardware.num_gpus_per_model == 0, (
            f"GPUs per model {self.config.hardware.num_gpus_per_model} does not divide total GPUs {total_gpus}"
        )
        self.model_comm_num_groups = (
            self.config.hardware.num_gpus_per_node
            * self.config.hardware.num_nodes
            // self.config.hardware.num_gpus_per_model
        )  # number of model communication groups
        LOGGER.debug(
            "Rank %d model communication group number %d, with local model communication group rank %d",
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
        )

        # Set the maximum rollout to be expected
        self.rollout = (
            self.config.training.rollout.max
            if self.config.training.rollout.epoch_increment > 0
            else self.config.training.rollout.start
        )

        # Set the training end date if not specified
        if hasattr(self.config.dataloader.training, "end") and self.config.dataloader.training.end is None:
            if not hasattr(self.config.dataloader.validation, "start"):
                excep_msg = "No validation start date specified, cannot dynamically set training end date."
                raise ValueError(excep_msg)
            LOGGER.info(
                "No end date specified for training data, setting default before validation start date %s.",
                self.config.dataloader.validation.start - 1,
            )
            self.config.dataloader.training.end = self.config.dataloader.validation.start - 1

        if not self.config.dataloader.get("pin_memory", True):
            LOGGER.info("Data loader memory pinning disabled.")

        self.check_dataset_slicing()

    def _check_resolution(self, resolution: str) -> None:
        assert (
            self.config.data.resolution.lower() == resolution.lower()
        ), f"Network resolution {self.config.data.resolution=} does not match dataset resolution {resolution=}"

    @cached_property
    def statistics(self) -> dict:
        return self.ds_train.statistics

    @cached_property
    def metadata(self) -> dict:
        return self.ds_train.metadata

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)

    @cached_property
    def timeincrement(self) -> int:
        """Determine the step size relative to the data frequency."""
        try:
            frequency = frequency_to_seconds(self.config.data.frequency)
        except ValueError as e:
            msg = f"Error in data frequency, {self.config.data.frequency}"
            raise ValueError(msg) from e

        try:
            timestep = frequency_to_seconds(self.config.data.timestep)
        except ValueError as e:
            msg = f"Error in timestep, {self.config.data.timestep}"
            raise ValueError(msg) from e

        assert timestep % frequency == 0, (
            f"Timestep ({self.config.data.timestep} == {timestep}) isn't a "
            f"multiple of data frequency ({self.config.data.frequency} == {frequency})."
        )

        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            timestep // frequency,
            frequency,
            timestep,
        )
        return timestep // frequency

    def check_dataset_slicing(self) -> None:
        """Check that the training, validation and test datasets do not overlap."""

        def get_overlap(range1: list[str], range2: list[str]) -> Union[list, None]:  # noqa: FA100
            intersection = set(range1) & set(range2)
            intersection = sorted(map(str, intersection))
            if len(intersection) > 6:
                intersection = intersection[:3] + ["..."] + intersection[-3:]
            return intersection

        if overlap := get_overlap(self.ds_train.dates, self.ds_valid.dates):
            excep_msg = "Training and validation data overlap, dates:"
            raise ValueError(excep_msg, overlap)
        if overlap := get_overlap(self.ds_train.dates, self.ds_test.dates):
            excep_msg = "Training and test data overlap, dates:"
            raise ValueError(excep_msg, overlap)
        if overlap := get_overlap(self.ds_valid.dates, self.ds_test.dates):
            excep_msg = "Validation and test data overlap, dates:"
            raise ValueError(excep_msg, overlap)

    @cached_property
    def ds_train(self) -> NativeGridDataset:
        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.training, resolve=True)),
            label="train",
        )

    @cached_property
    def ds_valid(self) -> NativeGridDataset:
        r = self.rollout
        r = max(r, self.config.dataloader.get("validation_rollout", 1))

        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.validation, resolve=True)),
            shuffle=False,
            rollout=r,
            label="validation",
        )

    @cached_property
    def ds_test(self) -> NativeGridDataset:

        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.test, resolve=True)),
            shuffle=False,
            label="test",
        )

    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        rollout: int = 1,
        label: str = "generic",
    ) -> NativeGridDataset:
        r = max(rollout, self.rollout)
        data = NativeGridDataset(
            data_reader=data_reader,
            rollout=r,
            multistep=self.config.training.multistep_input,
            timeincrement=self.timeincrement,
            model_comm_group_rank=self.model_comm_group_rank,
            model_comm_group_id=self.model_comm_group_id,
            model_comm_num_groups=self.model_comm_num_groups,
            shuffle=shuffle,
            label=label,
        )
        self._check_resolution(data.resolution)
        return data

    def _get_dataloader(self, ds: NativeGridDataset, stage: str) -> DataLoader:
        assert stage in {"training", "validation", "test"}
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            # number of worker processes
            num_workers=self.config.dataloader.num_workers[stage],
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=self.config.dataloader.get("pin_memory", True),
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")
