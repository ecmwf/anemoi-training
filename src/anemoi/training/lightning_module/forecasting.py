import logging
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterfaceForecasting

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
        super().__init__(config, graph_data, statistics, statistics_tendencies, data_indices, metadata, model_cls=AnemoiModelInterfaceForecasting)
        self.prediction_strategy = config.training.prediction_strategy
        self.step_functions = {
            "state": self._step_state,
            "residual": self._step_state,
            "tendency": self._step_tendency,
        }
        assert self.prediction_strategy in self.step_functions, f"Invalid prediction mode: {self.prediction_strategy}"
        if self.prediction_strategy == "tendency":
            assert statistics_tendencies is not None, "Tendency mode requires statistics_tendencies in dataset."
        LOGGER.info("Using prediction strategy: %s", self.prediction_strategy)


        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        
        train_loss = super().training_step(batch, batch_idx)

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
        use_checkpoint: bool = True,
        batch_target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, list]]:
        return self.step_functions[self.prediction_mode](batch, batch_idx, validation_mode, use_checkpoint, batch_target)


    def _step_state(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        batch_target: Optional[torch.Tensor] = None,
        use_checkpoint: bool = True,
        ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, list]]:
        
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.model.pre_processors_state(batch, in_place=False)  # normalized in-place
        
        if batch_target is not None:
            batch_target = self.model.pre_processors_state(batch_target, in_place=False)
        else:
            batch_target = batch

        # start rollout
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, inp_ens, multi_step, latlon, nvar)

        # Version 2 - which assume loss function takes in all the data at once. In this version we have to rollout and hold y_pred in memory
        y_preds = torch.zeros(batch.shape[0], self.rollout, *batch.shape[2:], device=self.device, dtype=batch.dtype)
        for rollout_step in range(self.rollout):
            y_pred = self(x)
            y_preds[:, rollout_step, ...] = y_pred
            x = self.advance_input(x, y_pred, batch, rollout_step)

        if use_checkpoint:  
            loss = checkpoint(self.loss, y_preds, batch_target[:, :, self.multi_step:, ..., self.data_indices.data.output.full], use_reentrant=False)
        else:
            loss = self.loss(y_preds, batch_target[:, :, self.multi_step:, ..., self.data_indices.data.output.full])

        loss *= 1.0 / self.rollout

        if validation_mode:
            dict_outputs = self.get_proc_and_unproc_data(y_preds, batch[:, : self.multi_step: self.multi_step + rollout_step, ..., self.data_indices.data.output.full])

            metrics = self.calculate_val_metrics(**dict_outputs)
    

        return loss, metrics, dict_outputs
    
    def _step_tendency(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        in_place_proc: bool = True,
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
            loss += checkpoint(self.loss, tendency_pred, tendency_target, use_reentrant=False)

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
    def get_proc_and_unproc_data(self, y_pred=None, y=None, y_pred_postprocessed=None, y_postprocessed=None):
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
                    
                # NOTE: This allows us to log metrics that operate along the time dimension  
                if m_value.shape[0] == y_pred.shape[2]: # check if it is a time-dependent metric
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
        x[:, -1, : , :, self.data_indices.model.input.prognostic] = y_pred[
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


class ForecastingLightningModuleICEnsemble(EnsembleCommunicationMixin, pl.LightningModule):
    """Ensemble version of Forecasting Lightning Module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ensemble_ic_generator = EnsembleInitialConditions(config=self.config, data_indices=self.data_indices)
        

    def _step(
        self,
        batch: tuple[Tensor, Tensor],  # shape (bs, multistep, latlon, nvars_full)
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
        x_ic = self.ensemble_ic_generator(x_center, x_ic)  # shape (bs, 

        if self.config.training.target_ic:
            x_target = x_ic
        else:
            x_target = x_center

        return super()._step(x_ic, batch_idx, validation_mode, x_target)
        # NOTE (rilwan-ade): If you want to target EDA, then this is where you change the 2nd arg, it should be x_ic

# NOTE (simon) - Which Communication Mixin is made to work with the Diffussion Model?
class ForecastingLightningModuleDiffussion(EnsembleCommunicationMixin, ForecastingLightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x: torch.Tensor, state_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.model(x, state_in, sigma, self.model_comm_group)

    # NOTE (rilwan-ade): what is the difference between redsidual and tendency?
    def _step_tendency(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        use_checkpoint: bool = True,
        **kwargs
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Forward pass of trainer for tendency prediction strategy.

        y_pred = model(x_t0)
        y_target = x_t1 - x_t0
        loss(y_pred, y_target)
        """
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}

        #TODO (rilwan-ade): Check if there is an ensemble dimension

        # Get batch tendencies from non processed batch
        batch_tendency_target = self.model.compute_processed_tendency(
            batch[:, self.multi_step : self.multi_step + self.rollout, ...],
            batch[:, self.multi_step - 1 : self.multi_step + self.rollout - 1, ...],
        )

        # state x is not processed)
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)
        # why do we need self.data_indices.data.input.full here? removes one variable it seems ...

        y_preds = []
        y_noiseds = []  # these are inital state + noised target tendency

        dict_outputs = defaultdict(list)

        for rollout_step in range(self.rollout):

            assert rollout_step == 0, "Diffusion model only supports single step training"

            # normalise inputs
            x_in = self.model.pre_processors_state(x, in_place=False, data_index=self.data_indices.data.input.full)

            # compute target tendency (normalised)
            tendency_target = batch_tendency_target[:, rollout_step]

            rnd_uniform = torch.rand(
                [tendency_target.shape[0], tendency_target.shape[1], 1, 1],
                device=tendency_target.device,
            )  # bs, ensemble size, latlon, nvar

            sigma = (
                self.model.sigma_max ** (1.0 / self.rho)
                + rnd_uniform * (self.model.sigma_min ** (1.0 / self.rho) - self.model.sigma_max ** (1.0 / self.rho))
            ) ** self.rho
            loss_weight = (sigma**2 + self.model.sigma_data**2) / (sigma * self.model.sigma_data) ** 2

            # prediction
            n = torch.randn_like(tendency_target) * sigma
            tendency_target_noised = tendency_target + n

            tendency_pred = self.model.fwd_with_preconditioning(
                tendency_target_noised,
                sigma,
                x_in,
                model_comm_group=self.model_comm_group,
            )

            # calculate loss
            # NOTE: currently only WeightedMSE implements calc and scale
            if use_checkpoint:
                l = checkpoint(self.loss.calc, tendency_pred, tendency_target, use_reentrant=False)
            else:
                l = self.loss.calc(tendency_pred, tendency_target)
            l = (l * loss_weight) # scaling by noise related weights
            l = self.loss.scale(l)
            loss += l

            # re-construct non-processed predicted state
            y_pred = self.model.add_tendency_to_state(x[:, -1, ...], tendency_pred)

            # advance input using non-processed x, y_pred and batch
            x = self.advance_input(x, y_pred, batch, rollout_step)

            if validation_mode:                
                y_preds.append(y_pred)
                y_noiseds.append(
                    self.model.add_tendency_to_state(x[:, -1, ...], tendency_target_noised)
                )

        
        if validation_mode:
            y_pred_postprocessed = torch.stack(y_preds, dim=1)
            y_postprocessed = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full] 

            
            y_noised = self.model.pre_processors_state(
                y_noiseds,
                in_place=False,
                data_index=self.data_indices.data.output.full,
            )
            
            # calculate_val_metrics requires processed inputs
            metrics = self.calculate_val_metrics(
                **dict_outputs
            )

            dict_outputs["y_noised"] = torch.split(y_noiseds, 1, dim=1)

        
            # dict_outputs["y_preds"] = torch.split(y_preds, 1, dim=1)
            

            # dict_outputs["y_postprocessed"] = torch.split(y_postprocessed, 1, dim=1)
            # dict_outputs["y_pred_postprocessed"] = torch.split(y_pred_postprocessed, 1, dim=1)

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics,  dict_outputs
