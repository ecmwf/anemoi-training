import math
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch import Tensor
from anemoi.models.data_indices.collection import IndexCollection
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import ModuleDict
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch import ModuleList

from anemoi.models.interface import AnemoiReconstructionModelInterface
from hydra.utils import instantiate

from anemoi.utils.config import DotDict
from anemoi.training.utils.jsonify import map_config_to_primitives

import logging
LOGGER = logging.getLogger(__name__)

class ReconstructionLightningModule(pl.LightningModule):
    """Graph reconstruction Lightning modules."""

    def __init__(
        self,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict,
        metadata: dict,
    ) -> None:
        """Initialize the graph VAE state.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        statistics : dict
            Statistics of the training data
        data_indices : dict
            Indices of the training data,
        metadata : dict
            Provenance information
        """
        super().__init__()

        # Load the graph data
        graph_data = graph_data.to(self.device)

        # Initialize the model
        self.model = AnemoiReconstructionModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=self.graph_data,
            config= DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        self.data_indices = data_indices

        self.save_hyperparameters()

        # LatLon and node weights
        self.latlons_data = graph_data[config.graph.data].x
        self.node_weights = graph_data[config.graph.data][config.model.node_loss_weight].squeeze()
        self.latlons_hidden = [graph_data[h_name].x for h_name in config.graph.hidden]
        self.latent_weights = graph_data[config.graph.hidden[-1]][config.model.node_loss_weight].squeeze()

        # Logger flag
        self.logger_enabled = (
            config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled
        )

        # Model configurations
        self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices)
        self.feature_weights = self.get_feature_weights(config, data_indices)
        self.multi_step = config.training.multistep_input

        # Loss and metrics initialization
        self.loss = instantiate(
            config.training.loss,
            config.training.loss_kwargs,
            config.training,
            node_weights=self.node_weights,
            latent_weights=self.latent_weights,
            feature_weights=self.feature_weights,
            data_indices_model_output=self.data_indices.model.output,
        )

        self.val_metrics = ModuleList(
            [
                instantiate(
                    vm_cfg,
                    node_weights=self.node_weights,
                    latent_weights=self.latent_weights,
                    feature_weights=self.feature_weights,
                    data_indices_model_output=self.data_indices.model.output,
                )
                for vm_cfg in config.training.val_metrics
            ],
        )

        # Save config and comm-related properties
        self.config = config
        self.num_gpus_per_model = config.hardware.num_gpus_per_model
        self.num_gpus_per_ensemble = config.hardware.num_gpus_per_ensemble
        
        # Setup communication groups (moved from init to Communications Mixins)
        self.setup_communication(config)

    @staticmethod
    def get_val_metric_ranges(config: DictConfig, data_indices: IndexCollection) -> dict:
        
        val_metric_ranges = defaultdict(list)

        for key, idx in data_indices.model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1:
                # Group metrics for pressure levels (e.g., Q, T, U, V, etc.)
                val_metric_ranges[f"pl_{split[0]}"].append(idx)

                if key in config.training.metrics or "all_individual" in config.training.metrics:
                    val_metric_ranges[key] = [idx]
            else:
                val_metric_ranges[f"sfc_{key}"].append(idx)
                val_metric_ranges[f"sfc"].append(idx)

            # Specific features to calculate metrics for    
            if key in config.training.metrics:
                val_metric_ranges[key] = [idx]

        if "all_grouped" in config.training.metrics:
            val_metric_ranges.update( 
                { "all" : [v for v in data_indices.model.output.name_to_index.values() ] }
            )
            
        return val_metric_ranges

    @staticmethod
    def get_feature_weights(config: DictConfig, data_indices):
        feature_weights = np.ones((len(data_indices.data.output.full),), dtype=np.float32) * config.training.feature_weights.default
        pressure_level = instantiate(config.training.pressure_level_scaler)

        LOGGER.info(
            "Pressure level scaling: use scaler %s with slope %.4f and minimum %.2f",
            type(pressure_level).__name__,
            pressure_level.slope,
            pressure_level.minimum,
        )

        for key, idx in data_indices.model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1:
                # Apply pressure level scaling
                if split[0] in config.training.feature_weights.pl:
                    feature_weights[idx] = config.training.feature_weights.pl[split[0]] * pressure_level.scaler(int(split[1]))
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                # Apply surface variable scaling
                if key in config.training.feature_weights.sfc:
                    feature_weights[idx] = config.training.feature_weights.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)

        feature_weights = torch.from_numpy(feature_weights)
        return feature_weights

    def training_step(self, batch: Tensor|tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """ run one training step.

        Args:
            batch: tuple
                Batch data. tuple of length 1 or 2.
                batch[0]: analysis, shape (bs, multi_step + rollout, nvar, latlon)
            batch_idx: int
                Training batch index
        """

        train_loss, _, y_preds = self._step(batch, batch_idx)

        # NOTE: The train loss is calculated on normalized data and the loss from each variable is scaled

        
        for loss_name, loss_value in train_loss.items():
            self.log(
                "train/loss/" + loss_name,
                loss_value,
                on_epoch=False,
                on_step=True,
                prog_bar=loss_name == self.loss.log_name,
                logger=self.logger_enabled,
                sync_dist=True,
            )
        


        return train_loss[self.loss.log_name]

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> None:
        """Generate the ensemble initial conditions and run one validation step.

        Args:
            batch: tuple
                Batch data. tuple of length 1 or 2.
                batch[0]: analysis, shape (bs, multi_step + rollout, nvar, latlon)
                batch[1] (optional): EDA perturbations, shape (multi_step, nens_per_device, nvar, latlon)
            batch_idx: int
                Validation batch index
        """

        with torch.no_grad():
            val_loss, metrics, dict_tensors = self._step(batch, batch_idx, validation_mode=True)

        for loss_name, loss_value in val_loss.items():
            self.log(
                f"val/loss/{self.loss.log_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=loss_name == self.loss.log_name,
                logger=self.logger_enabled,
                sync_dist=True,
            )


        for mname, mvalue in metrics.items():
            self.log(
                "val/" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch[0].shape[0],
                sync_dist=True,
            )

        return val_loss, dict_tensors

    def predict_step(self, batch: Tensor) -> Tensor:
        with torch.no_grad():
            batch = self.normalizer(batch, in_place=False)
            # add dummy ensemble dimension (of size 1)
            x = batch[:, None, ...]
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat.squeeze(dim=1), in_place=False)

    def _step(
        self,
        batch: Tensor,  # shape (bs, multistep, latlon, nvars_full)
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping, Tensor]:
        
        if self.config.training.ensemble_ic:
            x_center, x_ic = batch[0], batch[1]
            x_center_w_ic = self.ensemble_ic_generator(x_center, x_ic)  # shape (bs, nens_per_device, multistep, latlon, nvars_full)
            x = self.self.model.pre_processor_state(x_center_w_ic, inplace=False)  # normalized in-place shape = (bs, nens_per_device, multistep, latlon, input.full)
        else:
            x = self.self.model.pre_processor_state(batch, inplace=False)  # normalized in-place shape = (bs, nens_per_device, multistep, latlon, input.full)
        
        x_inp = batch[..., self.data_indices.data.input.full]


        metrics = {}

        x_rec, li_z_mu, z_logvar = self(x_inp)
        # reconstructed state and VAE latent space descriptors (needed for KL loss)\

        # TODO (rilwan-ade): when multi level latent space is implemented, need to update
        # the calculate_val_metrics handle multiple levels??(think about this) (can't use skip connections)
        z_mu = li_z_mu[0]

        x_target = x[
            ..., self.data_indices.data.output.full
        ]  # shape = (bs, multistep, latlon, output.full)

        loss: Tensor | dict = checkpoint(self.loss, x_rec, x_target, z_mu=z_mu, z_logvar=z_logvar, squash=True, use_reentrant=False)

        if validation_mode:
            dict_tensors = self.get_proc_and_unproc_data(x_rec, x_target, x_inp)
            metrics_new = self.calculate_val_metrics(x_rec, x_target, z_mu, z_logvar)

            metrics.update(metrics_new)

        # Reducing the time dimension out (leave it in)
        # TODO (rilwan-ade) make sure the z_logvar based evals handle the fact that there is a TIME dimension
        # TODO (rilwan-ade) make sure losses factor in this time and can report per time step in the sequence
        return loss, metrics, { **dict_tensors, "z_mu": z_mu[:, 0], "z_logvar": z_logvar[:, 0]}

    def get_proc_and_unproc_data(self, x_rec:Tensor, x_target:Tensor, x_inp:Tensor):
        
        x_rec_postprocessed = self.model.post_processors(x_rec, in_place=False, data_index=self.data_indices.data.output.full  )
        x_target_postprocessed = self.model.post_processors(x_target, in_place=False, data_index=self.data_indices.data.output.full )
        x_inp_postprocessed = self.model.post_processors(x_inp, in_place=False, data_index=self.data_indices.data.input.full )

        return {
            "x_rec": x_rec,
            "x_target": x_target,
            "x_inp": x_inp,
            "x_rec_postprocessed": x_rec_postprocessed,
            "x_target_postprocessed": x_target_postprocessed,
            "x_inp_postprocessed": x_inp_postprocessed,
        }

    def calculate_val_metrics(self, x_rec, x_target, x_inp, x_rec_postprocessed, x_target_postprocessed):
        metrics = {}


        for mkey, indices in self.val_metric_ranges.items():
            
            # for single metrics do no variable scaling and non processed data
            # TOOD (rilwan-ade): Update logging to use get_time_step and report lead time in hours
            if len(indices) == 1:
                metrics[f"{mkey}"] = self.metrics(x_rec_postprocessed[..., indices], x_target_postprocessed[..., indices], feature_scaling=False)
            else:
                # for metrics over groups of vars used preprocessed data to adjust for potentially differing ranges for each variable in the group
                metrics[f"{mkey}"] = self.metrics(x_rec[..., indices], x_target[..., indices], feature_scaling=False)

        return metrics
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x, self.model_comm_group)

    def on_train_start(self):
        if self._gather_matrix is None:
            self._gather_matrix = self._build_gather_matrix()  # only once

        effective_max_training_steps = self.effective_training_steps(adjust_for_accumulation=True)
        LOGGER.debug("Effective max training steps: %d", effective_max_training_steps)
        self.log(name="effective_max_training_steps", value=effective_max_training_steps, rank_zero_only=True)

    def on_validation_start(self):

        if self._gather_matrix is None:
            self._gather_matrix = self._build_gather_matrix()  # only once

    def lr_scheduler_step(self, scheduler, metric) -> None:
        scheduler.step(epoch=self.trainer.global_step)

    def configure_optimizers(self):
        lr = self.config.training.optimizer.rate
        if self.config.training.optimizer.scale_by_gpus:
            lr *= (
                config.hardware.num_nodes
                * config.hardware.num_gpus_per_node
                * config.training.optimizer.rate
                / config.hardware.num_gpus_per_model
            )

        if self.config.training.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=lr,
                weight_decay=self.config.training.optimizer.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                betas=(0.9, 0.95),
                lr=lr,
                weight_decay=self.config.training.optimizer.weight_decay,
            )  # , fused=True)

        # Scheduler
        # Calculate effective Max Steps
        if self.config.training.scheduler.iterations == "auto":
            iterations = self.training_steps()
        else:
            iterations = self.config.training.scheduler.iterations

        # Setting warmup steps
        if isinstance(self.config.training.scheduler.warmup_steps, float):
            assert self.config.training.scheduler.warmup_steps <= 1.0, "Warmup steps must be a float between 0 and 1"
            warmup_steps = int(self.config.training.scheduler.warmup_steps * iterations)
        else:
            warmup_steps = self.config.training.scheduler.warmup_steps
        
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.config.training.scheduler.lr_min,
            t_initial=iterations,
            warmup_t=warmup_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_steps(self) -> int:
        
        if self.config.training.max_steps is not None:
            training_steps = self.config.training.max_steps

        elif self.config.training.max_epochs is not None:
            train_batches_per_epoch = len(self.trainer.datamodule.ds_train) // self.config.dataloader.batch_size["training"]

            training_steps = self.config.training.max_epochs * train_batches_per_epoch

        return training_steps


        """Ensemble version of Reconstruction Lightning Module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)