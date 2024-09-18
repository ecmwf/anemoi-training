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
from anemoi.training.train.comm_mixins import DeterministicCommunicationMixin, EnsembleCommunicationMixin

LOGGER = logging.getLogger(__name__)

class ForecastingLightningModule(AnemoiLightningModule):
    def __init__(self, config, graph_data, statistics, statistics_tendencies, data_indices, metadata):
        super().__init__(config, graph_data, statistics, data_indices, metadata, model_cls=AnemoiForecastingModelInterface)
    
            self.step_functions = {
            "residual": self._step_residual,
            "tendency": self._step_tendency,
        }
        self.prediction_mode = "tendency" if self.model.tendency_mode else "residual"
        LOGGER.info("Using stepping mode: %s", self.prediction_mode)

        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

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


class ForecastingLightningModuleDeterministic(DeterministicCommunicationMixin, pl.LightningModule):
    """Deterministic version of Forecasting Lightning Module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ForecastingLightningModuleEnsemble(EnsembleCommunicationMixin, pl.LightningModule):
    """Ensemble version of Forecasting Lightning Module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)