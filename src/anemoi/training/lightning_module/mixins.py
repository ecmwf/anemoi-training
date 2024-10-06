

from functools import cached_property
import os
import math
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
import logging
from typing import Optional
from torch import Tensor
import torch

LOGGER = logging.getLogger(__name__)


class DeterministicCommunicationMixin:

    def setup_communication(self, config) -> None:
        self.model_comm_group = None

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model
        self.model_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_model,
        )

    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group


class EnsembleCommunicationMixin:
    """Mixin to handle model and ensemble communication groups."""

    def setup_communication(self, config) -> None:

        # Initialize variables for communication groups
        self.ens_comm_group: Optional[ProcessGroup] = None
        self.model_comm_group_size: int = 1
        self.ens_comm_group_size: int = 1
        self.model_comm_group_id: int = 0
        self.model_comm_group_rank: int = 0
        self.model_comm_num_groups: int = 1
        self.ens_comm_group_id: int = 0
        self.ens_comm_group_rank: int = 0
        self.ens_comm_num_groups: int = 1
        self.nens_per_device = self.config.training.ic_ensemble_size * self.config.training.noise_sample_per_ic

        """Setup communication groups for ensemble and model."""
        LOGGER.debug("Setting up communication groups...")

        # Get model communication group ID and rank
        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // self.config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % self.config.hardware.num_gpus_per_model

        # Assert that the number of GPUs per ensemble is divisible by the number of GPUs per model
        assert self.config.hardware.num_gpus_per_ensemble % self.config.hardware.num_gpus_per_model == 0, (
            "Invalid ensemble vs. model size GPU group configuration: "
            f"{self.config.hardware.num_gpus_per_ensemble} mod {self.config.hardware.num_gpus_per_model} != 0"
        )

        # Calculate the number of model and ensemble communication groups
        self.model_comm_num_groups = self.config.hardware.num_gpus_per_ensemble // self.config.hardware.num_gpus_per_model
        self.ens_comm_num_groups = math.ceil(
            self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes / self.config.hardware.num_gpus_per_ensemble,
        )

        # Get ensemble communication group ID and rank
        self.ens_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // self.config.hardware.num_gpus_per_ensemble
        self.ens_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % self.config.hardware.num_gpus_per_ensemble

        # Log debug info
        LOGGER.debug(
            "Model comm group ID = %d, rank = %d out of %d groups",
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.model_comm_num_groups,
        )
        LOGGER.debug(
            "Ensemble comm group ID = %d, rank = %d out of %d groups",
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
            self.ens_comm_num_groups,
        )

    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        """Set model communication group."""
        LOGGER.debug("Global rank %d: my model comms group: %s", self.global_rank, model_comm_group)
        self.model_comm_group = model_comm_group
        self.model_comm_group_size = dist.get_world_size(group=model_comm_group)

    def set_ensemble_comm_group(self, ens_comm_group: ProcessGroup) -> None:
        """Set ensemble communication group."""
        LOGGER.debug("Global rank %d: my ensemble comms group: %s", self.global_rank, ens_comm_group)
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_size = dist.get_world_size(group=ens_comm_group)

    @cached_property
    def _build_gather_matrix(self) -> Tensor:
        """Builds a matrix of shape (ens_comm_group_size * nens_per_device,
        num_model_groups * nens_per_device). This matrix is used to average the
        contributions of individual ensemble members gathered in the ensemble comm
        group. It accounts for duplicates and different model sharding communication
        groups, if applicable.

        E.g., suppose
            - nens_per_device = 3
            - ens_comm_group_size = 4
            - model_comm_group_size = 2 (i.e. 2 model comm groups, and a total of 6 unique ensemble members)
        Then the gather matrix has shape (12, 6) and looks like:
            - * ( 0.5 * eye(3)  0.5 * eye(3)         0           0        )^T
            - * (      0              0        0.5 * eye(3)  0.5 * eye(3) )
        """
        # sub-block used to average all contributions from a model comm group
        gather_matrix_block = (1.0 / self.model_comm_group_size) * torch.cat(
            [torch.eye(self.nens_per_device, dtype=self.dtype, device=self.device)] * self.model_comm_group_size, dim=1,
        )
        return torch.block_diag(*([gather_matrix_block] * self.model_comm_num_groups)).T
