import logging
import math
import os
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiForecastingModelInterface
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

LOGGER = logging.getLogger(__name__)

class ForecastingLightningModule(pl.LightningModule):
    """ Graph neural network forecaster for PyTorch Lightning."""

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

        self.model = AnemoiForecastingModelInterface(
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        # Flexible stepping function definition
        self.step_functions = {
            "residual": self._step_residual,
            "tendency": self._step_tendency,
        }
        self.prediction_mode = "tendency" if self.model.tendency_mode else "residual"
        LOGGER.info("Using stepping mode: %s", self.prediction_mode)

        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x
        self.node_weights = graph_data[config.graph.data][config.model.node_loss_weight].squeeze()

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled
        
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

        self.config = config
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

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
      
    def get_feature_weights(self, config: DictConfig, data_indices: IndexCollection ) -> torch.Tensor:
        """
        Calculates the feature weights for each output feature based on the configuration, data indices. User can specify weighting strategies based on pressure level, feature type, and inverse variance scaling. Any strategies provided are combined.

        Parameters
        ----------
        config (DictConfig): A configuration object that contains the training parameters.
        data_indices (IndexCollection): An object that contains the indices of the data.

        Returns
        -------
        torch.Tensor: A tensor that contains the calculates weights for the feature dimension during loss computation.

        """
        feature_weights = np.ones((len(data_indices.data.output.full),), dtype=np.float32) * config.training.feature_weighting.default
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
                if split[0] in config.training.feature_weighting.pl:
                    feature_weights[idx] = config.training.feature_weighting.pl[split[0]] * pressure_level.scaler(int(split[1]))
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                # Apply surface variable scaling
                if key in config.training.feature_weighting.sfc:
                    feature_weights[idx] = config.training.feature_weighting.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
        
        if config.training.feature_weighting.inverse_tendency_variance_scaling:
            variances = self.model.statistics_tendencies["stdev"][data_indices.data.output.full]
            feature_weights /= variances
            
        return torch.from_numpy(feature_weights)

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


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        #TODO: change this to use the log_name from the loss function to avoid hardcoding
        with torch.no_grad():
            val_loss, metrics, outputs = self._step(batch, batch_idx, validation_mode=True)
        self.log(
            f"val/loss/{self.loss.log_name}",
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
                "val/" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss, outputs

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        lead_time_to_eval: Optional[int] = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        return self.step_functions[self.prediction_mode](batch, batch_idx, validation_mode, lead_time_to_eval: Optional[int] = None)
    
    
    def get_proc_and_unproc_data(self, y_pred, y, y_pred_postprocessed=None, y_postprocessed=None):
        assert y_pred is not None or y_pred_postprocessed is not None, "Either y_pred or y_pred_postprocessed must be provided"
        assert y is not None or y_postprocessed is not None, "Either y or y_postprocessed must be provided"

        if y_postprocessed is None:
            y_postprocessed = self.model.post_processors_state(y, in_place=False)
        if y_pred_postprocessed is None:
            y_pred_postprocessed = self.model.post_processors_state(y_pred, in_place=False)
        
        if y_pred is None:
            y_pred = self.model.pre_processors_state(y_pred_postprocessed, in_place=False, data_index=self.data_indices.data.output.full)
        if y is None:
            y = self.model.pre_processors_state(y_postprocessed, in_place=False, data_index=self.data_indices.data.output.full)
        
        return {
            "y": y,
            "y_pred": y_pred,
            "y_postprocessed": y_postprocessed,
            "y_pred_postprocessed": y_pred_postprocessed,
        }


    def calculate_val_metrics(self, y_pred, y, rollout_step, y_pred_postprocessed, y_postprocessed):
        metrics = {}

        for mkey, indices in self.val_metric_ranges.items():
            
            # for single metrics do no variable scaling and non processed data
            # TOOD (rilwan-ade): Update logging to use get_time_step and report lead time in hours
            if len(indices) == 1:
                metrics[f"{mkey}_{rollout_step + 1}"] = self.metrics(y_pred_postprocessed[..., indices], y_postprocessed[..., indices], feature_scaling=False)
            else:
                # for metrics over groups of vars used preprocessed data to adjust for potentially differing ranges for each variable in the group
                metrics[f"{mkey}_{rollout_step + 1}"] = self.metrics(y_pred[..., indices], y[..., indices], feature_scaling=False)

        return metrics
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)


    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.model.input.prognostic] = y_pred[
            ...,
            self.data_indices.model.output.prognostic,
        ]

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            ...,
            self.data_indices.data.input.forcing,
        ]
        return x

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        lead_time_to_eval: Optional[int] = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        return self.step_functions[self.prediction_mode](batch, batch_idx, validation_mode, lead_time_to_eval: Optional[int] = None)
    
    
    def _step_residual(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        lead_time_to_eval: Optional[list[int]] = None
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.model.pre_processors_state(batch, in_place=False)  # normalized in-place
        metrics = {}

        # start rollout
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)

        outputs = defaultdict(list)
        for rollout_step in range(self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            # if rollout_step > 0: torch.cuda.empty_cache() # uncomment if rollout fails with OOM
            y_pred = self(x)

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)

            x = self.advance_input(x, y_pred, batch, rollout_step)

            if validation_mode:
                if lead_time_to_eval and rollout_step + 1 not in lead_time_to_eval:
                    continue
                dict_tensors = self.get_proc_and_unproc_data(y_pred, y) 
                metrics_next = self.calculate_val_metrics(**dict_tensors, rollout_step=rollout_step)
                metrics.update(metrics_next)
                
                for k in dict_tensors.keys():
                    outputs[k].append(outputs[k])

        # scale loss
        loss *= 1.0 / self.rollout

        return loss, metrics, outputs

    def _step_tendency(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        lead_time_to_eval: Optional[list[int]] = None
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}

        # x ( non-processed )
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)

        outputs = defaultdict(list)
        for rollout_step in range(self.rollout):

            # normalise inputs
            x_in = self.model.pre_processors_state(x, in_place=False, data_index=self.data_indices.data.input.full)

            # prediction (normalized tendency)
            tendency_pred = self(x_in)

            # re-construct non-processed predicted state
            y_pred = self.model.add_tendency_to_state(x[:, -1, ...], tendency_pred)

            # Target is full state
            y_target = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]

            # calculate loss
            loss += checkpoint(
                self.loss,
                self.model.pre_processors_state(y_pred, in_place=False, data_index=self.data_indices.data.output.full),
                self.model.pre_processors_state(y_target, in_place=False, data_index=self.data_indices.data.output.full),
                use_reentrant=False,
            )
            # TODO: We should try that too
            # loss += checkpoint(self.loss, y_pred, y_target, use_reentrant=False)

            # advance input using non-processed x, y_pred and batch
            x = self.advance_input(x, y_pred, batch, rollout_step)

            if validation_mode:
                if lead_time_to_eval and rollout_step + 1 not in lead_time_to_eval:
                    continue
                dict_tensors = self.get_proc_and_unproc_data(y_pred_postprocessed=y_pred,
                    y_postprocessed=y_target)
                metrics_next = self.calculate_val_metrics(
                    **dict_tensors,
                    rollout_step=rollout_step
                    
                )

                metrics.update(metrics_next)

                for k in dict_tensors.keys():
                    outputs[k].append(outputs[k])
                
                # Appending the inputs since alot of the validation callbacks seem to use it
                outputs["x"]

        # scale loss
        loss *= 1.0 / self.rollout


        return loss, metrics, outputs

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def configure_optimizers(self):
        if self.config.training.zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                betas=(0.9, 0.95),
                lr=self.config.training.optimizer.lr,
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