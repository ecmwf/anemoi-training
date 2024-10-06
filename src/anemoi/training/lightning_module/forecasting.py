import logging
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiForecastingModelInterface

from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.utils.checkpoint import checkpoint

from anemoi.training.lightning_module.mixins import DeterministicCommunicationMixin, EnsembleCommunicationMixin

from anemoi.training.lightning_module.anemoi import AnemoiLightningModule
from typing import Optional
LOGGER = logging.getLogger(__name__)
from torch import Tensor
from anemoi.training.data.inicond import EnsembleInitialConditions

class ForecastingLightningModule(AnemoiLightningModule):
    def __init__(self, config, graph_data, statistics, statistics_tendencies, data_indices, metadata):
        super().__init__(config, graph_data, statistics, statistics_tendencies,data_indices, metadata, model_cls=AnemoiForecastingModelInterface)
        self.step_functions = {
            "residual": self._step_residual,
            "tendency": self._step_tendency,
        }
        self.prediction_mode = "tendency" if self.model.tendency_mode else "residual"
        LOGGER.info("Using stepping mode: %s", self.prediction_mode)

        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        
        train_loss, _ = super().training_step(batch, batch_idx)

        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        batch_target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, list]]:
        return self.step_functions[self.prediction_mode](batch, batch_idx, validation_mode, batch_target)


    def _step_residual(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        batch_target: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, list]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.model.pre_processors_state(batch, in_place=False)  # normalized in-place
        if batch_target is not None:
            batch_target = self.model.pre_processors_state(batch_target, in_place=False)
        else:
            batch_target = batch

        # start rollout
        x = batch[:, :, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, inp_ens, multi_step, latlon, nvar)

        # Version 2 - which assume loss function takes in all the data at once. In this version we have to rollout and hold y_pred in memory
        y_preds = torch.zeros(batch.shape[0], self.rollout, *batch.shape[2:], device=self.device, dtype=batch.dtype)
        for rollout_step in range(self.rollout):
            y_pred = self(x)
            y_preds[:, rollout_step, ...] = y_pred
            x = self.advance_input(x, y_pred, batch, rollout_step)

        loss += checkpoint(self.loss, y_preds, batch_target[:, self.multi_step:, ..., self.data_indices.data.output.full], use_reentrant=False)
        loss *= 1.0 / self.rollout

        outputs = defaultdict(list)
        if validation_mode:
            dict_outputs = self.get_proc_and_unproc_data(y_preds, batch[:, : self.multi_step: self.multi_step + rollout_step, ..., self.data_indices.data.output.full])

            metrics = self.calculate_val_metrics(**dict_outputs)
    

        return loss, metrics, dict_outputs
    
    def _step_tendency(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        batch_target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, list]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}

        # x ( non-processed )
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)
        y_preds = torch.zeros(batch.shape[0], self.rollout, *batch.shape[2:], device=self.device, dtype=batch.dtype)
        outputs = defaultdict(list)
        for rollout_step in range(self.rollout):

            # normalise inputs
            x_in = self.model.pre_processors_state(x, in_place=False, data_index=self.data_indices.data.input.full)

            # prediction (normalized tendency)
            tendency_pred = self(x_in)

            # re-construct non-processed predicted state
            y_pred = self.model.add_tendency_to_state(x[:, :, -1, ...], tendency_pred)
            y_preds[:, :, rollout_step, ...] = y_pred
            # advance input using non-processed x, y_pred and batch
            x = self.advance_input(x, y_pred, batch, rollout_step)

        # calculate loss
        batch_target = batch_target if batch_target is not None else batch

        y_target = batch_target[:, :, self.multi_step: self.multi_step + rollout_step, ..., self.data_indices.data.output.full]
        loss += checkpoint(
            self.loss,
            self.model.pre_processors_state(y_pred, in_place=False, data_index=self.data_indices.data.output.full),
            self.model.pre_processors_state(y_target, in_place=False, data_index=self.data_indices.data.output.full),
            use_reentrant=False,
        )
            # TODO: We should try that too
            # loss += checkpoint(self.loss, y_pred, y_target, use_reentrant=False)
        loss *= 1.0 / self.rollout
            

        if validation_mode:
            dict_outputs = self.get_proc_and_unproc_data(y_preds, batch[:, self.multi_step: self.multi_step + rollout_step, ..., self.data_indices.data.output.full])

            metrics = self.calculate_val_metrics(**dict_outputs)

        # scale loss
        loss *= 1.0 / self.rollout

        return loss, metrics, dict_outputs

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

    def calculate_val_metrics(self, y_pred, y, y_pred_postprocessed, y_postprocessed):
        """
        Calculate validation metrics.

        Parameters
        ----------
        y_pred : torch.Tensor (bs, ens_pred, timesteps, latlon, nvar)
            Predicted output tensor.
        y : torch.Tensor (bs, ens_target, timesteps, latlon, nvar)
            Target output tensor.
        y_pred_postprocessed : torch.Tensor (bs, end_pred, timesteps, latlon, nvar)
            Postprocessed predicted output tensor.
        y_postprocessed : torch.Tensor (bs, end_target, timesteps, latlon, nvar)
            Postprocessed target output tensor.

        Returns
        -------
        dict[str, torch.Tensor]
        
        """
        metric_vals = {}
        timesteps = y_pred.shape[2]

        for metric in self.val_metrics:
            for mkey, indices in self.val_metric_ranges.items():

                # for single metrics do no variable scaling and non processed data
                # TOOD (rilwan-ade): Update logging to use get_time_step and report lead time in hours
                
                if len(indices) == 1:
                    _args = (y_pred_postprocessed[..., indices], y_postprocessed[..., indices])
                else:
                    _args = (y_pred[..., indices], y[..., indices])

                m_value = metric(*_args, feature_scaling=False, squash=(-2, -1))  # squash spatial and feature dims
                    
                # determining if metric has squashed in time dimension
                if m_value.shape[0] == y_pred.shape[2]:
                    for i in range(timesteps):
                        metric_vals[f"{mkey}_{i + 1}"] = m_value[i]
                else:
                    metric_vals[f"{mkey}"] = m_value

        return metric_vals

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, :, -1, :, self.data_indices.model.input.prognostic] = y_pred[
            ...,
            self.data_indices.model.output.prognostic,
        ]

        # get new "constants" needed for time-varying fields
        x[:, :, -1, :, self.data_indices.model.input.forcing] = batch[
            :,
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



class ForecastingLightningModuleICEnsemble(EnsembleCommunicationMixin, pl.LightningModule):
    """Ensemble version of Forecasting Lightning Module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ensemble_ic_generator = EnsembleInitialConditions(config=self.config, data_indices=self.data_indices)
        

    def _step(
        self,
        batch: list[Tensor],  # shape (bs, multistep, latlon, nvars_full)
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping, Tensor]:

        """Run one  step.

        Args:
            batch: tuple
                Batch data..
                batch[0]: analysis, shape (bs, timesteps, nvar, latlon)
                batch[1] (optional): EDA perturbations, shape (timesteps, nens_input, nvar, latlon)
            batch_idx: int
                batch index   
            validation_mode: bool
        """
        x_center, x_ic = batch[0], batch[1]
        
        # First we generate the ensemble of initial conditions
        x_ic = self.ensemble_ic_generator(x_center, x_ic)  # shape (bs, nens_input, timesteps, latlon, nvars_full)

        batch_inp = self.model.pre_processor_state(x_ic, inplace=False)  # shape = (bs, nens_input, multistep, latlon, input.full)
        batch_target_center = self.model.pre_processor_state(x_center, inplace=False)  # shape = (bs, nens_target, multistep, latlon, 
        batch_target_ic = self.model.pre_processor_state(x_ic, inplace=False)  # shape = (bs, nens_target, multistep, latlon, 

        x_inp = batch_inp[..., self.data_indices.data.input.full]
        x_target_ic = batch_target_ic[..., self.data_indices.data.output.full] # shape = (bs,1, multistep, latlon, output.full)
        x_target_center = batch_target_center[..., self.data_indices.data.output.full] # shape = (bs,ens_ic, multistep, latlon, output.full)

        if self.config.training.target_ic:
            x_target = x_target_ic
        else:
            x_target = x_target_center

        return super()._step(x_ic, batch_idx, validation_mode, x_target)
        # #NOTE (rilwan-ade): If you want to target EDA, then this is where you change the 2nd arg, it should be x_ic


