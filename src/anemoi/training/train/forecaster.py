# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import math
import os
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.training.losses.mse import WeightedMSELoss
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask

LOGGER = logging.getLogger(__name__)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        statistics_tendencies : dict
            Statistics of the training data tendencies
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information

        """
        super().__init__()

        graph_data = graph_data.to(self.device)

        self.model = AnemoiModelInterface(
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        # Prediction strategy can be either residual, state or tendency
        self.prediction_strategy = config.training.prediction_strategy
        self.step_functions = {
            "state": self._step_state,
            "residual": self._step_residual,
            "tendency": self._step_tendency,
        }
        assert self.prediction_strategy in self.step_functions, f"Invalid prediction mode: {self.prediction_strategy}"
        if self.prediction_strategy == "tendency":
            assert statistics_tendencies is not None, "Tendency mode requires statistics_tendencies in dataset."
        LOGGER.info("Using prediction strategy: %s", self.prediction_strategy)

        self.data_indices = data_indices
        self.sigma_delta = statistics_tendencies["stdev"][self.data_indices.data.output.full][
            self.data_indices.model.output.prognostic
        ]
        self.sigma_state = (
            1
            / self.pre_processors_state.processors.state.normalizer._norm_mul[self.data_indices.model.input.prognostic]
        )

        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x
        self.node_weights = graph_data[config.graph.data][config.model.node_loss_weight].squeeze()

        if config.model.get("output_mask", None) is not None:
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()
        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        self.metric_ranges, self.metric_ranges_validation, self.feature_weights = self.metrics_loss_scaling(
            config,
            data_indices,
        )
        self.loss = WeightedMSELoss(node_weights=self.node_weights, feature_weights=self.feature_weights)
        self.metrics = WeightedMSELoss(node_weights=self.node_weights, feature_weights=None, ignore_nans=True)

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input

        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.use_zero_optimizer = config.training.zero_optimizer

        self.model_comm_group = None

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.enable_plot = config.diagnostics.plot.enabled

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model
        self.model_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)

    def metrics_loss_scaling(self, config: DictConfig, data_indices: IndexCollection) -> tuple[dict, torch.Tensor]:
        metric_ranges = defaultdict(list)
        metric_ranges_validation = defaultdict(list)
        loss_scaling = (
            np.ones((len(data_indices.data.output.full),), dtype=np.float32) * config.training.feature_weighting.default
        )

        pressure_level = instantiate(config.training.pressure_level_scaler)

        LOGGER.info(
            "Pressure level scaling: use scaler %s with slope %.4f and minimum %.2f",
            type(pressure_level).__name__,
            pressure_level.slope,
            pressure_level.minimum,
        )

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges[f"pl_{split[0]}"].append(idx)
                # Create pressure levels in loss scaling vector
                if split[0] in config.training.feature_weighting.pl:
                    loss_scaling[idx] = config.training.feature_weighting.pl[split[0]] * pressure_level.scaler(
                        int(split[-1]),
                    )
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                metric_ranges[f"sfc_{key}"].append(idx)
                # Create surface variables in loss scaling vector
                if key in config.training.feature_weighting.sfc:
                    loss_scaling[idx] = config.training.feature_weighting.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges[key] = [idx]
        loss_scaling = torch.from_numpy(loss_scaling)
        # metric for validation, after postprocessing
        for key, idx in data_indices.model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1 and split[1].isdigit():
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges_validation[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges_validation[f"sfc_{key}"].append(idx)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges_validation[key] = [idx]
        return metric_ranges, metric_ranges_validation, loss_scaling

    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.internal_model.input.prognostic] = y_pred[
            ...,
            self.data_indices.internal_model.output.prognostic,
        ]

        x[:, -1] = self.output_mask.rollout_boundary(x[:, -1], batch[:, -1], self.data_indices)

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.internal_model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.internal_data.input.forcing,
        ]
        return x

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        in_place_proc: bool = True,
        use_checkpoint: bool = True,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        return self.step_functions[self.prediction_strategy](
            batch,
            batch_idx,
            validation_mode,
            in_place_proc,
            use_checkpoint,
        )

    def _step_state(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        in_place_proc: bool = True,
        use_checkpoint: bool = True,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Forward pass of trainer for state and residual prediction strategy."""
        del batch_idx, in_place_proc
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.model.pre_processors_state(batch, in_place=not validation_mode)  # normalized in-place
        metrics = {}

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)

        y_preds = []
        for rollout_step in range(self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x)

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.internal_data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss

            if use_checkpoint:
                loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)
            else:
                loss += self.loss(y_pred, y)

            x = self.advance_input(x, y_pred, batch, rollout_step)

            if validation_mode:
                metrics_next, y_preds_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    rollout_step,
                    enable_plot=self.enable_plot,
                )
                metrics.update(metrics_next)
                y_preds.extend(y_preds_next)

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    def _step_residual(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        in_place_proc: bool = True,
        use_checkpoint: bool = True,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Forward pass of trainer for residual prediction strategy.

        y_pred = model(x_t0)
        y_target = x_t1 - x_t0
        loss(norm(y_pred - x_t0), norm(y_target))
        """
        del batch_idx, in_place_proc
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.model.pre_processors_state(batch, in_place=not validation_mode)  # normalized in-place
        metrics = {}

        # Get batch tendencies from non processed batch
        batch_residual_target = self.compute_target_residual(
            batch[:, self.multi_step : self.multi_step + self.rollout, ...],
            batch[:, self.multi_step - 1 : self.multi_step + self.rollout - 1, ...],
        )

        # state x is not processed)
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)

        y_preds = []
        for rollout_step in range(self.rollout):
            # normalise inputs
            # prediction (normalized tendency)
            y_pred = self(x)
            y_pred_residual = self.compute_target_residual(y_pred, x)

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.internal_data.output.full]
            y_residual = batch_residual_target[:, rollout_step]

            # access to normalizers only here?
            # Just normalise the tendencies for the normalised states as they enter the loss? ###
            y_pred_residual = self.norm_delta(y_pred_residual, self.sigma_state, self.sigma_delta)
            y_residual = self.norm_delta(y_residual, self.sigma_state, self.sigma_delta)

            # calculate loss
            if use_checkpoint:
                loss += checkpoint(self.loss, y_pred_residual, y_residual, use_reentrant=False)
            else:
                loss += self.loss(y_pred_residual, y_residual)

            # re-construct non-processed predicted state

            # advance input using non-processed x, y_pred and batch
            x = self.advance_input(x, y_pred, batch, rollout_step)

            if validation_mode:
                metrics_next, y_preds_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    rollout_step,
                    enable_plot=self.enable_plot,
                )
                metrics.update(metrics_next)
                y_preds.extend(y_preds_next)
        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    def _step_tendency(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        in_place_proc: bool = True,
        use_checkpoint: bool = True,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Forward pass of trainer for tendency prediction strategy.

        y_pred = model(x_t0)
        y_target = x_t1 - x_t0
        loss(y_pred, y_target)
        """
        del batch_idx, in_place_proc
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}

        # Get batch tendencies from non processed batch
        batch_tendency_target = self.compute_target_tendency(
            batch[:, self.multi_step : self.multi_step + self.rollout, ...],
            batch[:, self.multi_step - 1 : self.multi_step + self.rollout - 1, ...],
        )

        # state x is not processed)
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)

        y_preds = []
        for rollout_step in range(self.rollout):

            # normalise inputs
            x_in = self.model.pre_processors_state(x, in_place=False, data_index=self.data_indices.data.input.full)

            # prediction (normalized tendency)
            tendency_pred = self(x_in)

            tendency_target = batch_tendency_target[:, rollout_step]

            # calculate loss
            if use_checkpoint:
                loss += checkpoint(self.loss, tendency_pred, tendency_target, use_reentrant=False)
            else:
                loss += self.loss(tendency_pred, tendency_target)

            # re-construct non-processed predicted state
            y_pred = self.model.add_tendency_to_state(x[:, -1, ...], tendency_pred)

            # advance input using non-processed x, y_pred and batch
            x = self.advance_input(x, y_pred, batch, rollout_step)

            if validation_mode:
                y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]

                # calculate_val_metrics requires processed inputs
                metrics_next, _ = self.calculate_val_metrics(
                    None,
                    None,
                    rollout_step,
                    self.enable_plot,
                    y_pred_postprocessed=y_pred,
                    y_postprocessed=y,
                )

                metrics.update(metrics_next)

                y_preds.extend(
                    self.model.pre_processors_state(
                        y_pred,
                        in_place=False,
                        data_index=self.data_indices.data.output.full,
                    ),
                )
        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    def compute_target_tendency(self, x_t1: torch.Tensor, x_t0: torch.Tensor) -> torch.Tensor:
        tendency = self.model.pre_processors_tendency(
            x_t1[..., self.data_indices.data.output.full] - x_t0[..., self.data_indices.data.output.full],
            in_place=False,
            data_index=self.data_indices.data.output.full,
        )
        # diagnostic variables are taken from x_t1, normalised as full fields:
        tendency[..., self.data_indices.model.output.diagnostic] = self.model.pre_processors_state(
            x_t1[..., self.data_indices.data.output.diagnostic],
            in_place=False,
            data_index=self.data_indices.data.output.diagnostic,
        )
        return tendency

    def compute_target_residual(self, x_t1: torch.Tensor, x_t0: torch.Tensor) -> torch.Tensor:
        tendency = x_t1[..., self.data_indices.data.output.full] - x_t0[..., self.data_indices.data.output.full]
        # diagnostic variables are taken from x_t1, normalised as full fields:
        tendency[..., self.data_indices.model.output.diagnostic] = x_t1[..., self.data_indices.data.output.diagnostic]
        return tendency

    def norm_delta(self, y_delta: torch.Tensor, sigma_state: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        return sigma_state * y_delta / sigma_delta

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
        enable_plot: bool = False,
        y_pred_postprocessed: torch.Tensor = None,
        y_postprocessed: torch.Tensor = None,
    ) -> tuple[dict, list]:
        metrics = {}
        y_preds = []
        if y_postprocessed is None:
            y_postprocessed = self.model.post_processors_state(y, in_place=False)
        if y_pred_postprocessed is None:
            y_pred_postprocessed = self.model.post_processors_state(y_pred, in_place=False)

        for mkey, indices in self.metric_ranges_validation.items():
            metrics[f"{mkey}_{rollout_step + 1}"] = self.metrics(
                y_pred_postprocessed[..., indices],
                y_postprocessed[..., indices],
            )

        if enable_plot:
            y_preds.append(y_pred)
        return metrics, y_preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            "train_wmse",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)
        self.log(
            "val_wmse",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss, y_preds

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        if self.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=self.lr,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                betas=(0.9, 0.95),
                lr=self.lr,
            )  # , fused=True)

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
