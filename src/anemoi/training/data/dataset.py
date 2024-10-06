# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
import os
import random
from functools import cached_property
from typing import Callable
from typing import Optional
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
from einops import rearrange
from anemoi.training.utils.seeding import get_base_seed

LOGGER = logging.getLogger(__name__)


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        rollout: int = 0,
        multistep: int = 1,
        timeincrement: int = 1,
        timestep: str = "6h",
        model_comm_group_rank: int = 0,
        model_comm_group_id: int = 0,
        model_comm_num_groups: int = 1,
        model_comm_group_nworkers: int = 1,
        shuffle: bool = True,
        label: str = "generic",
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        rollout : int, optional
            length of rollout window, by default 12
        timeincrement : int, optional
            time increment between samples, by default 1
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        model_comm_group_rank : int, optional
            process rank in the torch.distributed group (important when running on multiple GPUs), by default 0
        model_comm_group_id: int, optional
            device group ID, default 0
        model_comm_num_groups : int, optional
            total number of device groups, by default 1
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"

        """
        self.label = label

        self.data = data_reader

        self.rollout = rollout
        self.timeincrement = timeincrement
        self.timestep = timestep

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # DDP-relevant info
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_nworkers = model_comm_group_nworkers
        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return dataset tendency statistics."""
        # The statistics_tendencies are lazily loaded
        if callable(self.data.statistics_tendencies):
            self.data.statistics_tendencies = self.data.statistics_tendencies(self.timestep)
        return self.data.statistics_tendencies

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id

        # Total number of valid ICs is dataset length minus rollout minus additional multistep inputs
        len_corrected = len(self.data) - (self.rollout + (self.multi_step - 1)) * self.timeincrement

        # Divide this equally across shards (one shard per group!)
        shard_size = len_corrected // self.model_comm_num_groups
        shard_start = self.model_comm_group_id * shard_size + (self.multi_step - 1) * self.timeincrement
        shard_end = min((self.model_comm_group_id + 1) * shard_size, len(self.data) - self.rollout * self.timeincrement)

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)

        LOGGER.debug(
            "Worker %d (pid %d, global_rank %d, model comm group %d) has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        # should we check lightning env variable "PL_SEED_WORKERS" here?
        # but we alwyas want to seed these anyways ...

        base_seed = get_base_seed()

        seed = (
            base_seed * (self.model_comm_group_id + 1) - worker_id
        )  # note that test, validation etc. datasets get same seed
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.debug(
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "group_rank %d, base_seed %d) using seed %d, sanity rnd %f"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            base_seed,
            seed,
            sanity_rnd,
        )

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over the dataset.

        The datasets are retrieved by Anemoi Datasets from zarr files. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. 
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.chunk_index_range,
                size=self.n_samples_per_worker,
                replace=False,
            )
        else:
            shuffled_chunk_indices = self.chunk_index_range

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement

            x = self.data[start : end : self.timeincrement]
            x = rearrange(x, "dates variables ensemble gridpoints -> ensemble dates gridpoints variables")
            self.ensemble_dim = 0

            yield torch.from_numpy(x)

    def __len__(self) -> int:
        # Calculation for __len__ of dataset should align to calculation used in per_worker_init

        len_corrected = len(self.data) - (self.rollout + (self.multi_step - 1)) * self.timeincrement
        # Divide this equally across shards (one shard per group!)
        shard_size = len_corrected // self.model_comm_num_groups
        shard_start = self.model_comm_group_id * shard_size + (self.multi_step - 1) * self.timeincrement
        shard_end = min((self.model_comm_group_id + 1) * shard_size, len(self.data) - self.rollout * self.timeincrement)
        samples = shard_end - shard_start

        return (samples // self.model_comm_group_nworkers) * self.model_comm_group_nworkers

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Rollout: {self.rollout}
            Multistep: {self.multi_step}
            Timeincrement: {self.timeincrement}
        """


class EnsNativeGridDataset(NativeGridDataset):
    """Iterable ensemble dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        comm_group_rank: int = 0,
        comm_group_id: int = 0,
        comm_num_groups: int = 1,
        shuffle: bool = True,
        label: str = "generic",
        ens_members_per_device: int = 1,
        num_gpus_per_ens: int = 1,
        num_gpus_per_model: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        rollout : int, optional
            length of rollout window, by default 12
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        ens_members_per_device: int, optional
            number of ensemble members input for each GPU device, by default 1
        comm_group_rank : int, optional
            process rank in the torch.distributed group (important when running on multiple GPUs), by default 0
        comm_group_id: int, optional
            device group ID, default 0
        comm_num_groups : int, optional
            total number of device groups, by default 1
        shuffle : bool, optional
            Shuffle batches, by default True

        Raises
        ------
        RuntimeError
            Multistep value cannot be negative.
        """
        super().__init__(
            data_reader=data_reader,
            rollout=rollout,
            multistep=multistep,
            timeincrement=timeincrement,
            model_comm_group_rank=comm_group_rank,
            model_comm_group_id=comm_group_id,
            model_comm_num_groups=comm_num_groups,
            shuffle=shuffle,
            label=label,
        )

        self._seed: Optional[int] = None
        self._worker_id: Optional[int] = None

        self.comm_group_id = comm_group_id
        self.comm_group_rank = comm_group_rank

        # Lazy init
        self.ens_members_per_device = ens_members_per_device
        self.num_gpus_per_ens = num_gpus_per_ens
        self.num_gpus_per_model = num_gpus_per_model

    @property
    def num_eda_members(self) -> int:
        """Return number of EDA members."""
        return self.data.shape[2] - 1

    @property
    def eda_flag(self) -> bool:
        """Return whether EDA is enabled."""
        return self.data.shape[2] > 1

    def sample_eda_members(self, num_eda_members: int = 9) -> np.ndarray:
        """Subselect EDA ensemble members assigned to the current device."""
        tot_ens = self.ens_members_per_device * self.num_gpus_per_ens // self.num_gpus_per_model

        assert tot_ens <= num_eda_members, f"Can't generate an ensemble of size {tot_ens} from {num_eda_members} EDA perturbations"

        eda_member_gen_idx = self.rng.choice(range(num_eda_members), size=tot_ens, replace=False)
        offset = 1  # index=0 analysis, index=1 EDA recentred
        eda_member_gen_idx += offset

        effective_rank = self.comm_group_rank // self.num_gpus_per_model
        eda_member_idx = np.sort(
            eda_member_gen_idx[effective_rank * self.ens_members_per_device : self.ens_members_per_device * (1 + effective_rank)],
        )

        LOGGER.debug(
            "GPU with global rank %s, Worker id %s, comm_group_id %s, comm_group_rank %s will receive EDA member(s) %s",
            self.global_rank,
            self._worker_id,
            self.comm_group_id,
            self.comm_group_rank,
            eda_member_gen_idx,
        )

        return eda_member_gen_idx, eda_member_idx

    def __iter__(self):
        """Return an iterator over the dataset.

        The datasets are retrieved by ECML Tools from zarr files. This iterator yields
        chunked batches for DDP and sharded training.
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(self.chunk_index_range, size=self.n_samples_per_worker, replace=False)
        else:
            shuffled_chunk_indices = self.chunk_index_range

        for i in shuffled_chunk_indices:
            # start and end time indices, for analysis and EDA
            start = i - (self.multi_step - 1) * self.timeincrement
            end_an = i + (self.rollout + 1) * self.timeincrement
            end_eda = i + self.timeincrement

            if self.eda_flag:
                eda_member_gen_idx, eda_member_idx = self.sample_eda_members(self.num_eda_members)
            else:
                eda_member_gen_idx = None
                eda_member_idx = None

            x_an = self.data[start : end_an : self.timeincrement, :, 0:1, ...]
            x_an = rearrange(torch.from_numpy(x_an), "dates variables ensemble gridpoints -> ensemble dates gridpoints variables")

            x_pert: Optional[torch.Tensor] = None
            if self.eda_flag:

                x_pert = self.data[start : end_eda : self.timeincrement, eda_member_idx,...]
                sample = (
                    x_an,
                    rearrange(
                        torch.from_numpy(x_pert),
                        "dates variables ensemble gridpoints -> ensemble dates gridpoints variables",
                    ),
                )
            else:
                sample = (x_an,)

            # Handle debug logging
            self._log_debug_info(
                self._worker_id,
                os.getpid(),
                self.global_rank,
                self.comm_group_id,
                self.comm_group_rank,
                start,
                end_an,
                self.eda_flag,
                eda_member_gen_idx,
                eda_member_idx,
                analysis_shape=list(sample[0].shape),
                perturbation_shape=list(sample[1].shape) if len(sample) > 1 else "n/a",
            )

            yield sample

    def __repr__(self) -> str:
        return (
            f"""
            {super().__repr__()}
            Dataset: {self.data}
            Rollout: {self.rollout}
            Multistep: {self.multi_step}
            Timeincrement: {self.timeincrement}
            EDA: {self.eda_flag}
            """
            f"""
            Number of EDA members:
            {self.num_eda_members}" if self.eda_flag else
            """
        )

    def _log_debug_info(
        self,
        worker_id: Optional[int],
        pid: int,
        global_rank: Optional[int],
        comm_group_id: Optional[int],
        comm_group_rank: Optional[int],
        tstart: int,
        tend_an: int,
        eda_flag: bool,
        eda_member_gen_idx: Optional[np.ndarray],
        eda_member_idx: Optional[np.ndarray],
        analysis_shape: list,
        perturbation_shape: list,
    ):
        if eda_flag:
            eda_debug_args = (str(eda_member_gen_idx.tolist()), str(eda_member_idx.tolist()))
            eda_message_part = ", eda_mem_gen = %s, eda_members = %s"
        else:
            eda_debug_args = ()
            eda_message_part = ""

        base_message = (
            "Worker %d (pid %d, global_rank %d, comm group %d, group_rank %d) got [tstart, tend_an) = [%d, %d)"
            + eda_message_part
            + " ..."
        )

        LOGGER.debug(
            base_message,
            worker_id,
            pid,
            global_rank,
            comm_group_id,
            comm_group_rank,
            tstart,
            tend_an,
            *eda_debug_args,  # This will only be added if eda_debug_args is not empty
        )

        LOGGER.debug(
            "Sample shapes: analysis = %s, EDA perturbation = %s",
            analysis_shape,
            perturbation_shape,
        )


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process.

    Calls WeatherBenchDataset.per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID

    Raises
    ------
    RuntimeError
        If worker_info is None

    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )
