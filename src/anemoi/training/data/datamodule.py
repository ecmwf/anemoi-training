# © 2023 Anemoi Inc. All rights reserved.
import logging
import os
from functools import cached_property
from typing import Callable

import pytorch_lightning as pl
from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import abc

from anemoi.training.data.dataset import NativeGridDataset, EnsNativeGridDataset, worker_init_func

LOGGER = logging.getLogger(__name__)


class AnemoiBaseDataModule(pl.LightningDataModule):
    """Base class for Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        frequency = self.config.data.frequency
        timestep = self.config.data.timestep
        assert (
            isinstance(frequency, str) and isinstance(timestep, str) and frequency[-1] == "h" and timestep[-1] == "h"
        ), f"Error in format of timestep, {timestep}, or data frequency, {frequency}"
        assert (
            int(timestep[:-1]) % int(frequency[:-1]) == 0
        ), f"Timestep isn't a multiple of data frequency, {timestep}, or data frequency, {frequency}"
        self.timeincrement = int(timestep[:-1]) // int(frequency[:-1])
        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            self.timeincrement,
            frequency,
            timestep,
        )

        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))  # global rank
        self.model_comm_group_id = (
            self.global_rank // self.config.hardware.num_gpus_per_model
        )  # id of the model communication group
        self.model_comm_group_rank = (
            self.global_rank % self.config.hardware.num_gpus_per_model
        )  # rank within one model communication group
        total_gpus = self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes
        assert total_gpus % self.config.hardware.num_gpus_per_model == 0, (
            f"GPUs per model {self.config.hardware.num_gpus_per_model} does not divide total GPUs {total_gpus}"
        )
        self.model_comm_num_groups = (
            total_gpus // self.config.hardware.num_gpus_per_model
        )  # number of model communication groups

    def _check_resolution(self, resolution: str) -> None:
        assert (
            self.config.data.resolution.lower() == resolution.lower()
        ), f"Network resolution {self.config.data.resolution=} does not match dataset resolution {resolution=}"

    @cached_property
    def statistics(self) -> dict:
        return self.ds_train.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        if self.config.training.tendency_mode or self.config.training.feature_weighting.inverse_tendency_variance_scaling:
            return self.ds_train.statistics_tendencies
        return None

    @cached_property
    def metadata(self) -> dict:
        return self.ds_train.metadata

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)
    
    @abc.abstractmethod
    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        label: str = "generic",
    ) -> NativeGridDataset:
        """
        Get the dataset.
        This method must be implemented by child classes.
        """
        pass

    def _get_dataloader(self, ds: NativeGridDataset, stage: str) -> DataLoader:
        assert stage in {"training", "validation", "test"}
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            num_workers=self.config.dataloader.num_workers[stage],
            pin_memory=True,
            worker_init_fn=worker_init_func,
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")

class AnemoiForecastingDataModule(AnemoiBaseDataModule):
    """Data module for forecasting models, with rollout functionality."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.rollout = (
            self.config.training.rollout.max
            if self.config.training.rollout.epoch_increment > 0
            else self.config.training.rollout.start
        )

    @cached_property
    def ds_train(self) -> NativeGridDataset:
        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.training, resolve=True)),
            label="train",
            rollout=self.rollout,
        )

    @cached_property
    def ds_valid(self) -> NativeGridDataset:
        r = self.rollout
        if self.config.diagnostics.eval.enabled:
            r = max(r, self.config.diagnostics.eval.rollout)
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
            rollout=self.rollout,
            label="test",
        )

class AnemoiEnsForecastingDataModule(AnemoiForecastingDataModule):
    """Anemoi Ensemble data module for PyTorch Lightning."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize Anemoi data module.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        """
        super().__init__(config=config)
        self.nens_per_device = self.config.training.ic_ensemble_size * self.config.training.noise_sample_per_ic
        self.ens_comm_group_id = (
            self.global_rank // self.config.hardware.num_gpus_per_ensemble
        )  # id of the ensemble communication group the rank is participating in
        self.ens_comm_group_rank = (
            self.global_rank % self.config.hardware.num_gpus_per_ensemble
        )  # rank within one ensemble communication group

        total_gpus = self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes

        assert (
            total_gpus
        ) % self.config.hardware.num_gpus_per_ensemble == 0, (
            f"GPUs per ensemble {self.config.hardware.num_gpus_per_ensemble} does not divide total GPUs {total_gpus}"
        )

        self.ens_comm_num_groups = (
            self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes // self.config.hardware.num_gpus_per_ensemble
        )  # number of ensemble communication groups
        LOGGER.debug(
            "Rank %d ensemble communication group number %d, with local group rank %d",
            self.global_rank,
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
        )


    def _get_dataset(
        self,
        data_reader,
        shuffle: bool = True,
        rollout: int = 1,
        label: str = "generic",
    ) -> EnsNativeGridDataset:

        r = max(rollout, self.rollout)

        data = EnsNativeGridDataset(
            data_reader=data_reader,
            rollout=r,
            multistep=self.config.training.multistep_input,
            timeincrement=self.timeincrement,
            comm_group_rank=self.ens_comm_group_rank,
            comm_group_id=self.ens_comm_group_id,
            comm_num_groups=self.ens_comm_num_groups,
            shuffle=shuffle,
            label=label,
            ens_members_per_device=self.nens_per_device,
            num_gpus_per_ens=self.config.hardware.num_gpus_per_ensemble,
            num_gpus_per_model=self.config.hardware.num_gpus_per_model,
        )

        self._check_resolution(data.resolution)
        return data

class AnemoiReconstructionDataModule(AnemoiBaseDataModule):
    """Data module for reconstruction tasks, without rollout functionality."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    @cached_property
    def ds_train(self) -> NativeGridDataset:
        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.training, resolve=True)),
            label="train",
        )

    @cached_property
    def ds_valid(self) -> NativeGridDataset:
        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.validation, resolve=True)),
            shuffle=False,
            label="validation",
            
        )

    @cached_property
    def ds_test(self) -> NativeGridDataset:
        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.test, resolve=True)),
            shuffle=False,
            label="test"
        )

    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        label: str = "generic",
    ) -> NativeGridDataset:
        data = NativeGridDataset(
            data_reader=data_reader,
            rollout=0,
            multistep=self.config.training.multistep_input,
            timeincrement=self.timeincrement,
            timestep=self.config.data.timestep,
            model_comm_group_rank=self.model_comm_group_rank,
            model_comm_group_id=self.model_comm_group_id,
            model_comm_num_groups=self.model_comm_num_groups,
            model_comm_group_nworkers=self.config.dataloader.num_workers[label],
            shuffle=shuffle,
            label=label,
        )
        self._check_resolution(data.resolution)
        return data

class AnemoiEnsReconstructionDataModule(AnemoiReconstructionDataModule):
    """Data module for reconstruction tasks, without rollout functionality."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.nens_per_device = self.config.training.ic_ensemble_size * self.config.training.noise_sample_per_ic
        self.ens_comm_group_id = (
            self.global_rank // self.config.hardware.num_gpus_per_ensemble
        )  # id of the ensemble communication group the rank is participating in
        self.ens_comm_group_rank = (
            self.global_rank % self.config.hardware.num_gpus_per_ensemble
        )  # rank within one ensemble communication group

        total_gpus = self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes

        assert (
            total_gpus
        ) % self.config.hardware.num_gpus_per_ensemble == 0, (
            f"GPUs per ensemble {self.config.hardware.num_gpus_per_ensemble} does not divide total GPUs {total_gpus}"
        )

        self.ens_comm_num_groups = (
            self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes // self.config.hardware.num_gpus_per_ensemble
        )  # number of ensemble communication groups
        LOGGER.debug(
            "Rank %d ensemble communication group number %d, with local group rank %d",
            self.global_rank,
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
        )


    def _get_dataset(
        self,
        data_reader,
        shuffle: bool = True,
        label: str = "generic",
    ) -> EnsNativeGridDataset:


        data = EnsNativeGridDataset(
            data_reader=data_reader,
            rollout=0,
            multistep=self.config.training.multistep_input,
            timeincrement=self.timeincrement,
            comm_group_rank=self.ens_comm_group_rank,
            comm_group_id=self.ens_comm_group_id,
            comm_num_groups=self.ens_comm_num_groups,
            shuffle=shuffle,
            label=label,
            ens_members_per_device=self.config.training.nens_per_device,
            num_gpus_per_ens=self.config.hardware.num_gpus_per_ensemble,
            num_gpus_per_model=self.config.hardware.num_gpus_per_model,
        )

        self._check_resolution(data.resolution)
        return data