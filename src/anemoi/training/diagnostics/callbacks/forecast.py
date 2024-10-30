import csv
import os
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist

from omegaconf import DictConfig

from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks import ParentUUIDCallback

from anemoi.training.diagnostics.plots.plots import init_plot_settings, plot_loss_map
from anemoi.training.diagnostics.plots.plots import plot_graph_features
from anemoi.training.diagnostics.plots.plots import plot_histogram
from anemoi.training.diagnostics.plots.plots import plot_loss
from anemoi.training.diagnostics.plots.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots.plots import plot_predicted_multilevel_flat_sample
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback

# function should take
from functools import lru_cache
from functools import wraps

from torch import Tensor
from pytorch_lightning.callbacks import LearningRateMonitor
from anemoi.training.diagnostics.callbacks import GraphTrainableFeaturesPlot, BasePlotCallback
from anemoi.training.diagnostics.callbacks import BaseLossMapPlot, BaseLossBarPlot
from anemoi.training.diagnostics.callbacks import WeightGradOutputLoggerCallback

from anemoi.training.diagnostics.callbacks import safe_cast_to_numpy
from anemoi.training.diagnostics.callbacks import get_time_step, increment_to_hours, generate_time_steps

from anemoi.training.losses.utils import get_monitored_metric_name
import logging
from anemoi.training.diagnostics.callbacks.common_callbacks import get_common_callbacks

LOGGER = logging.getLogger(__name__)

from pytorch_lightning.utilities.rank_zero import rank_zero_only


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window.

    This differs from the RolloutEval in the following ways:
       - ensures that all validation plots are based on output of RolloutEval instead of normal eval
       - reduces the amount of callbacks that hook to the on validation batch end func
       - simplifies some logic and reduces redundancies
    """

    def __init__(self, config, val_dset_len, callbacks: list[Callback]) -> None:
        """Initialize RolloutEval callback.

            # Some Rollout Forecasts rely on the output from this class - Those callbacks are passed in to 
            the constructor and use the output of this class
            # In this way this reduces the doubling of code where each callback may re-implement the denormalization of the output

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings
        """
        super().__init__()

        LOGGER.debug(
            "Setting up RolloutEval callback with rollout = %d, frequency = %d ...",
            config.diagnostics.metrics.rollout_eval.rollout,
            config.diagnostics.metrics.rollout_eval.frequency,
        )

        self.rollout = config.diagnostics.metrics.rollout_eval.rollout
        self.frequency = config.diagnostics.eval.frequency
        self.eval_frequency = BasePlotCallback.get_eval_frequency(self, config.diagnostics.metrics.rollout_eval.frequency, val_dset_len)
        self.lead_time_to_eval = config.diagnostics.eval.lead_time_to_eval
        
        self.callbacks_validation_batch_end = [
            cb for cb in callbacks 
            if 'on_validation_batch_end' in cb.__class__.__dict__
        ]
        self.callbacks_validation_epoch_end = [
            cb for cb in callbacks 
            if 'on_validation_epoch_end' in cb.__class__.__dict__
        ]

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        # TODO: Move this around
        # - Under this set up the plot eval frequency needs to be a multiple of the eval frequency
        # - Or you assure that forward rollout loop can occur either for a plotting or an eval
        # - Or use some approximation which finds the closest eval step to the plot step and uses that

        if self.op_on_this_batch(batch_idx):
            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = torch.autocast(device_type=batch.device.type, dtype=dtype) if dtype is not None else nullcontext()

            with context:
                with torch.no_grad():
                    outputs = self._eval(
                        pl_module, batch[0],
                        batch_idx
                    )  # Ignores the predictions from the validation loop

                    for cb in self.callbacks_validation_batch_end:
                        cb.on_validation_batch_end(
                                trainer, pl_module, outputs, batch, batch_idx)
                            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # TODO (rilwan-ade): Add setting to allow frequency setting to apply to callbacks that operate on a epoch frequency to e.g. per two / four validation epochs
        for callback in self.callbacks_validation_epoch_end:
            callback.on_validation_epoch_end(trainer, pl_module)

    def _eval(self, pl_module: pl.LightningModule, batch: torch.Tensor, batch_idx: int) -> None:
        """Rolls out the model and calculates the validation metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            Lightning module object
        batch: torch.Tensor # shape = (bs, ens, input_steps + forecast_steps, latlon, nvar)
            Batch tensor
        batch_idx: int
            Batch index
        """
        
        loss, metrics, outputs = pl_module._step(batch, validation_mode=True, 
                                                 batch_idx=batch_idx,
                                                 lead_time_to_eval=self.lead_time_to_eval)

        self._log(pl_module, loss, metrics, batch.shape[0])

        # Appending the non processed inputs since most of callbacks use it
        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + self.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ]
        data = safe_cast_to_numpy(self.post_processors_state(input_tensor))
        outputs["data"] = data

        return outputs
    

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:
        # This is the validation loss using the training loss function, averaged across all rollouts
        train_loss_name = pl_module.loss.name

        rollout_step_str_start = get_time_step(self.time_step, 1)
        rollout_step_str_end = get_time_step(self.time_step, self.rollout + 1)

        pl_module.log(
            f"rval/loss/{pl_module.loss.name}/r_{rollout_step_str_start}_to_{rollout_step_str_end}_freq_{self.frequency}",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            batch_size=bs,
            sync_dist=True,
            rank_zero_only=True,
        )
        
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"rval/{mname}",
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=pl_module.logger_enabled,
                batch_size=bs,
                sync_dist=True,
                rank_zero_only=True,
            )
    
    def op_on_this_batch(self, batch_idx ):
        callbacks_idx_to_run = [
            idx
            for idx, pcallback in enumerate(self.callbacks_validation_batch_end)
            if pcallback.op_on_this_batch(batch_idx)
        ]  # A boolean list indicating which callbacks require the eval step
                
        return (((batch_idx + 1) % self.eval_frequency) == 0) or len(callbacks_idx_to_run) > 0

class ForecastingLossBarPlot(BaseLossBarPlot):
    """Plots the accumulated forecasting unsqueezed loss over rollouts."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.counter = 0
        self.dict_rstep_loss_map = dict()
        self.time_step = config.data.timestep
        self.lead_time_to_eval = config.diagnostics.eval.lead_time_to_eval
        # TODO - this lead_time_to_eval is dependent on the value set in RolloutEval class - need to change this to factor that

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int) -> None:
        if self.op_on_this_batch(batch_idx):
            self.accumulate(trainer, pl_module, outputs, batch)
            self.counter += 1

    def accumulate(self, trainer, pl_module, outputs, batch) -> None:
        LOGGER.debug("Parameters to plot: %s", self.config.diagnostics.plot.parameters)
        
        for idx in range(len(outputs["y_pred"])):
            loss_ = pl_module.loss(
                outputs["y_pred"][idx],
                outputs["y"][idx],
                squash=False
            ).sum(0)  # (nvar)

            key = f"r{get_time_step(self.time_step, self.lead_time_to_eval[idx])}"

            if key not in self.dict_rstep_loss_map:
                self.dict_rstep_loss_map[key] = loss_
            else:
                self.dict_rstep_loss_map[key] += loss_

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.plot(trainer, pl_module)

        keys = list(self.dict_rstep_loss_map.keys())
        for k in keys:
            del self.dict_rstep_loss_map[k]
        self.dict_rstep_loss_map = dict()

        self.counter = 0
        torch.cuda.empty_cache()

    def _plot(self, trainer, pl_module) -> None:
        LOGGER.debug("Parameters to plot: %s", self.config.diagnostics.plot.parameters)

        parameter_names = list(pl_module.data_indices.model.output.name_to_index.keys())
        paramter_positions = list(pl_module.data_indices.model.output.name_to_index.values())
        # reorder parameter_names by position
        self.parameter_names = [parameter_names[i] for i in np.argsort(paramter_positions)]

        for key in self.dict_rstep_loss_map:
            
            loss_avg = (
                self.dict_rstep_loss_map[key] / self.counter
            )

            dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.AVG)

            if pl_module.global_rank == 0:
                sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group

                fig = plot_loss(safe_cast_to_numpy(loss_avg[sort_by_parameter_group], colors, xticks, legend_patches))
                
                epoch = trainer.current_epoch
                
                self._output_figure(
                    pl_module.logger,
                    fig,
                    tag=f"val_barplot_{pl_module.loss.name}_r{key}_{epoch:03d}_gstep{pl_module.global_step}",
                    exp_log_tag=f"val_bar_{pl_module.loss.name}_r{key}_{epoch:03d}_gstep{pl_module.global_step}",
                )

class PlotSample(BasePlotCallback):
    """Plots a post-processed sample: input, target and prediction."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the PlotSample callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        self.lead_time_to_eval = self.config.diagnostics.eval.lead_time_to_eval

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        # Build dictionary of indices and parameters to be plotted
        diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
            for name in self.config.diagnostics.plot.parameters
        }

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        data = outputs["data"]

        
        for idx in range(len(outputs["y_pred"])):

            rollout_step = self.lead_time_to_eval[idx]

            fig = plot_predicted_multilevel_flat_sample(
                plot_parameters_dict,
                self.config.diagnostics.plot.per_sample,
                self.latlons,
                self.config.diagnostics.plot.accumulation_levels_plot,
                self.config.diagnostics.plot.cmap_accumulation,
                data[0, ...].squeeze(),
                data[rollout_step + 1, ...].squeeze(),
                # output_tensor[rollout_step, ...],
                safe_cast_to_numpy(outputs['y_pred_postprocessed'][rollout_step])
            )

            key = f"r{get_time_step(self.time_step, self.lead_time_to_eval[idx])}"

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_sample_{key}_epoch_{epoch:03d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_sample_{key}_rank{local_rank:01d}",
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if self.op_on_this_batch(batch_idx):
            self.plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)

class PlotHistograms(BasePlotCallback):
    """Plots histogram metrics comparing target and prediction."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the PlotHistograms callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        """
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        self.lead_time_to_eval = config.diagnostics.eval.lead_time_to_eval

    @rank_zero_only
    def _plot_histogram(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        data = outputs["data"]
        output_tensor = outputs['y_pred_postprocessed']

        # for rollout_step in range(pl_module.rollout):
        for idx in range(len(outputs["y_pred"])):
            if self.config.diagnostics.plot.parameters_histogram is None:
                continue
            rollout_step = self.lead_time_to_eval[idx]
            diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
            plot_parameters_dict_histogram = {
                pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
                for name in self.config.diagnostics.plot.parameters_histogram
            }

            fig = plot_histogram(
                plot_parameters_dict_histogram,
                data[0, ...].squeeze(),
                data[rollout_step + 1, ...].squeeze(),
                # output_tensor[rollout_step, ...],
                safe_cast_to_numpy(output_tensor[idx]),
            )

            key = f"r{get_time_step(self.time_step, self.lead_time_to_eval[idx])}"
            self._output_figure(
                logger,
                fig,
                tag=f"gnn_pred_val_histo_{key}_epoch_{epoch:03d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_histo_{key}_{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if self.op_on_this_batch(batch_idx):
            self._plot_histogram(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)

class PlotPowerSpectrum(BasePlotCallback):
    """Plots power spectrum metrics comparing target and prediction."""

    def __init__(self, config, val_dset_len, **kwargs):
        """Initialise the PlotPowerSpectrum callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        """
        super().__init__(config, op_on_batch=True, val_dset_len=val_dset_len, **kwargs)
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        self.lead_time_to_eval = config.diagnostics.eval.lead_time_to_eval

    @rank_zero_only
    def _plot_spectrum(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())

        data = outputs["data"]
        output_tensor = outputs['y_pred_postprocessed']

        # for rollout_step in range(pl_module.rollout):
        for idx in range(len(outputs["y_pred"])):
            rollout_step = self.lead_time_to_eval[idx]
            if self.config.diagnostics.plot.parameters_spectrum is not None:
                diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
                plot_parameters_dict_spectrum = {
                    pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
                    for name in self.config.diagnostics.plot.parameters_spectrum
                }

                fig = plot_power_spectrum(
                    plot_parameters_dict_spectrum,
                    self.latlons,
                    data[0, ...].squeeze(),
                    data[rollout_step + 1, ...].squeeze(),
                    # output_tensor[rollout_step, ...],
                    safe_cast_to_numpy(output_tensor[idx]),
                )

                self._output_figure(
                    logger,
                    fig,
                    tag=f"gnn_pred_val_spec_rstep_{rollout_step:02d}_epoch_{epoch:03d}_batch{batch_idx:04d}_rank0",
                    exp_log_tag=f"val_pred_spec_rstep_{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if self.op_on_this_batch(batch_idx):
            self._plot_spectrum(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)

class ForecastingLossMapPlot(BaseLossMapPlot):
    def __init__(self, config, val_dset_len, **kwargs) -> None:
        super().__init__(config, val_dset_len, **kwargs)
        self.dict_rstep_loss_map = dict()
        self.lead_time_to_eval = config.diagnostics.eval.lead_time_to_eval

    def accumulate(self, trainer, pl_module, outputs) -> None:
        
        for idx in range(len(outputs["y_pred_postprocessed"])):
            rollout_step = self.lead_time_to_eval[idx]
            # if self.log_rstep is True or (rollout_step + 1 in self.log_rstep):
            preds = outputs["y_pred_postprocessed"][idx]
            y_pred_postprocessed = outputs["y_pred_postprocessed"][idx]

            # TODO: change so squash=False also doesn't squash in the spatial dimension
            loss_map = pl_module.loss(
                preds, y_pred_postprocessed, squash=False ) # (latlon, nvar)

            key = f"r{get_time_step(self.config.data.timestep, rollout_step + 1)}"
            if key not in self.dict_rstep_loss_map:
                self.dict_rstep_loss_map[key] = loss_map
            else:
                self.dict_rstep_loss_map[key] += loss_map

    def _plot(self, trainer, pl_module) -> None:
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.cpu().numpy())
        
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: name for name in self.config.diagnostics.plot.parameters
        }
            
        keys = list(self.dict_rstep_loss_map.keys())
        for key in keys:
            loss_map = self.dict_rstep_loss_map[key] / self.counter
            dist.reduce(loss_map, dst=0, op=dist.ReduceOp.AVG)

            if pl_module.global_rank == 0:

                # Plot loss maps for individual parameters
                fig = plot_loss_map(
                    plot_parameters_dict,
                    self.latlons,
                    safe_cast_to_numpy(loss_map),
                )
                fig.tight_layout()
                self._output_figure(trainer, fig, f"val_map_r{key}_epoch{trainer.current_epoch:03d}_gstep{trainer.global_step:06d}", f"val_loss_r{key}_{trainer.current_epoch:03d}")

                # Plot loss map for all parameters
                fig, self.inner_subplots_kwargs = plot_loss_map(
                    {0: "all"},
                    safe_cast_to_numpy(np.rad2deg(pl_module.latlons_hidden.clone())),
                    safe_cast_to_numpy(loss_map.mean(dim=-1, keepdim=True))
                )


    def reset(self) -> None:
        self.dict_rstep_loss_map = dict()

class AnemoiCheckpointRollout(AnemoiCheckpoint):
    #TODO (rilwan-ade): Need to Debug this class & write tests for it
    """A checkpoint callback that saves the model after every validation epoch and
    manages directories based on rollout changes."""

    def __init__(self, name: str, config: DictConfig, increment_on: str, **kwargs):
        super().__init__(name, config, **kwargs)
        self.curr_rollout_steps = None
        self.dir_path_base = Path(self.dirpath)

        # Add an option to change the monitor to track the rollout specific loss or the average loss over all rollout steps
        self.timestep = config.data.timestep
        # self.rollout = next(config.training.rollout.schedule.values())

        assert increment_on in [
            "training_epoch",
            "validation_epoch",
        ], f"increment_on must be either 'training_epoch' or 'validation_epoch'. Got {increment_on}"
        self.increment_on = increment_on

    def on_fit_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        rollout_step_str = get_time_step(self.timestep, pl_module.rollout)
        self.dirpath = self.dir_path_base / f"r{rollout_step_str}"
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)

        self.curr_rollout_steps = pl_module.rollout
        assert (
            self.curr_rollout_steps is not None
        ), "Rollout steps must be defined before training. Please set `rollout` in the LightningModule."

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.increment_on == "training_epoch":
            # current_rollout_value = pl_module.rollout  # assuming `rollout_value` is a property of the module
            if self.curr_rollout_steps is not None and self.curr_rollout_steps != pl_module.rollout:
                self._reset_checkpoint_directory(trainer, pl_module.rollout)
            self.curr_rollout_steps = pl_module.rollout
            super().on_train_epoch_end(trainer, pl_module)

    # def on_validation_model_train(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.increment_on == "validation_epoch":
            # current_rollout_value = pl_module.rollout  # assuming `rollout_value` is a property of the module
            if self.curr_rollout_steps is not None and self.curr_rollout_steps != pl_module.rollout:
                self._reset_checkpoint_directory(trainer, pl_module.rollout)
            self.curr_rollout_steps = pl_module.rollout

            super().on_validation_end(trainer, pl_module)

        elif self.increment_on == "training_epoch":
            super().on_validation_end(trainer, pl_module)

    def _reset_checkpoint_directory(self, trainer: Trainer, rollout_value):
        LOGGER.info("Rollout value changed. Resetting checkpoint directory.")

        # Dump the current best_k_models to a CSV file
        if trainer.is_global_zero and self.best_k_models:
            current_dirpath = Path(self.dirpath)
            csv_file_path = current_dirpath / "best_k_models.csv"
            with open(csv_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Checkpoint Path", "Score"])
                for checkpoint_path, score in sorted(
                    self.best_k_models.items(), key=lambda item: item[1], reverse=(self.mode == "max")
                ):
                    writer.writerow([checkpoint_path, score.item()])

            LOGGER.info("Dumped best_k_models for rollout value %d to %s", rollout_value, csv_file_path)

        # Update the directory for checkpoints based on the rollout value
        rollout_step_str = get_time_step(self.timestep, rollout_value)

        self.dirpath = self.dir_path_base / f"r{rollout_step_str}"
        self.best_k_models = {}
        self.best_model_path = ""
        self.best_model_score = None
        self.kth_best_model_path = ""
        self.kth_value = torch.tensor(float("inf") if self.mode == "min" else -float("inf"))

        if trainer.is_global_zero:
            # Create new directory if it does not exist
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {"dir_path_base": str(self.dir_path_base), "curr_rollout_steps": self.curr_rollout_steps, "timestep": self.timestep}
        )

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # super().load_state_dict(state_dict)

        self.name = state_dict.get("name", self.name)
        self.config = state_dict.get("config", self.config)
        self.start = state_dict.get("start", self.start)

        self.best_model_score = state_dict["best_model_score"]
        self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
        self.kth_value = state_dict.get("kth_value", self.kth_value)
        self.best_k_models = state_dict.get("best_k_models", self.best_k_models)
        self.last_model_path = state_dict.get("last_model_path", self.last_model_path)

        self.dir_path_base = Path(state_dict.get("dir_path_base", self.dir_path_base))
        self.best_model_path = state_dict["best_model_path"]
        self.dirpath = state_dict["dirpath"]
        self.curr_rollout_steps = state_dict.get("curr_rollout_steps", self.curr_rollout_steps)

class EarlyStoppingRollout(EarlyStopping):
    #TODO (rilwan-ade): Need to Debug this class & write tests for it
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
        timestep: str = "6h",
    ):
        super().__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            strict,
            check_finite,
            stopping_threshold,
            divergence_threshold,
            check_on_train_epoch_end,
            log_rank_zero_only,
        )
        self.curr_rollout_steps = None
        self.timestep = timestep

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.curr_rollout_steps is not None and self.curr_rollout_steps != pl_module.rollout:
            self._reset_early_stopping()
        self.curr_rollout_steps = pl_module.rollout
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.curr_rollout_steps is not None and self.curr_rollout_steps != pl_module.rollout:
            self._reset_early_stopping()
        self.curr_rollout_steps = pl_module.rollout
        super().on_validation_end(trainer, pl_module)

    def _reset_early_stopping(self):
        rank_zero_info("Rollout value changed. Resetting early stopping.")
        self.wait_count = 0
        torch_inf = torch.tensor(torch.inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def _improvement_message(self, current: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the
        monitored score."""
        rollout_steps_str = get_time_step(self.timestep, self.curr_rollout_steps)
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f} @ rollout={rollout_steps_str}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f} @ rollout={rollout_steps_str}"
        return msg

class RolloutScheduler(Callback):
    def __init__(self, schedule: dict[int, int]):
        super().__init__()

        self.schedule = schedule
        self.last_observed_step = None
        self._rollout = None

        LOGGER.info(str(self))

    @property
    def current_rollout(self) -> int:
        return self._rollout

    def calculate_rollout(self, step: int) -> int:
        applicable_keys = [key for key in self.schedule.keys() if int(key) <= step]
        if not applicable_keys:
            return min(self.schedule.values(), default=0, key=lambda x: int(x))
        max_applicable_key = max(applicable_keys, key=lambda x: int(x))
        return int(self.schedule[max_applicable_key])

    def update_rollout(self, step: int):
        r = self.calculate_rollout(step)
        if r != self._rollout:
            self._rollout = r
            LOGGER.debug(f"Rollout set to {self._rollout} at step {step}")
            self.last_observed_step = step

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.update_rollout(pl_module.global_step)
        pl_module.rollout = self._rollout

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        self.update_rollout(pl_module.global_step)
        pl_module.rollout = self._rollout

    def state_dict(self) -> dict:
        return {
            "schedule": self.schedule,
            "last_observed_step": self.last_observed_step,
            "_rollout": self._rollout,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.schedule = state_dict["schedule"]
        self.last_observed_step = state_dict["last_observed_step"]
        self._rollout = state_dict["_rollout"]

    def __str__(self) -> str:
        return f"RolloutScheduler(schedule={self.schedule})"



def setup_rollout_eval_callbacks(config, val_dset_len):
    """Create and return rollout evaluation callbacks."""
    callbacks = []

    if config.diagnostics.plot.plot_sample:
        callbacks.append(PlotSample(config, val_dset_len))

    if config.diagnostics.plot.loss_map:
        callbacks.append(ForecastingLossMapPlot(config, val_dset_len))

    if config.diagnostics.plot.loss_bar:
        callbacks.append(ForecastingLossBarPlot(config, val_dset_len))

    if config.diagnostics.plot.plot_spectral_loss:
        callbacks.append(PlotPowerSpectrum(config, val_dset_len))

    rollout_eval = RolloutEval(
        config=config,
        val_dset_len=val_dset_len,
        callbacks=callbacks
    )
    return rollout_eval

def get_callbacks(config: DictConfig, monitored_metrics, val_dset_len) -> list:
    trainer_callbacks = get_common_callbacks(config, monitored_metrics)
    
    # Add forecast-specific callbacks
    rollout_eval = setup_rollout_eval_callbacks(config, val_dset_len)
    trainer_callbacks.append(rollout_eval)
    
    return trainer_callbacks

# def get_callbacks(config: DictConfig, monitored_metrics, val_dset_len) -> list:
#     """Setup callbacks for PyTorch Lightning trainer.

#     Parameters
#     ----------
#     config : DictConfig
#         Job configuration
#     monitored_metrics : list
#         List of monitored metrics to track during training

#     Returns
#     -------
#     list
#         A list of PyTorch Lightning callbacks
#     """
#     trainer_callbacks = []

#     def setup_checkpoint_callbacks():
#         """Create and return checkpoint-related callbacks."""
#         ckpt_callbacks = []
#         if config.diagnostics.profiler:
#             LOGGER.warning("Profiling is enabled - no checkpoints will be written!")
#             return []

#         checkpoint_configs = config.diagnostics.checkpoints

#         for ckpt_cfg in checkpoint_configs:
#             filename, mode, dirpath = None, "max", None

#             if ckpt_cfg.type == 'interval':
#                 dirpath = os.path.join(
#                     config.hardware.paths.checkpoints,
#                     next(k for k in ("every_n_train_steps", "train_time_interval", "every_n_epochs") if ckpt_cfg.get(k))
#                 )

#             elif ckpt_cfg.type == "performance":
#                 OmegaConf.set_readonly(ckpt_cfg, False)
#                 ckpt_cfg.kwargs['monitor'] = get_monitored_metric_name(monitored_metrics, ckpt_cfg['monitor'])
#                 OmegaConf.set_readonly(ckpt_cfg, True)
#                 monitor_name = f"perf_{ckpt_cfg.kwargs['monitor'].replace('/', '_')}"
#                 dirpath = os.path.join(config.hardware.paths.checkpoints, f"perf_{monitor_name}")
#                 filename = f"epoch={{epoch}}-step={{step}}-{monitor_name}-{{{ckpt_cfg.kwargs['monitor']}}:.5f}"

#             ckpt_callbacks.append(
#                 AnemoiCheckpointRollout(
#                     config=config,
#                     filename=filename,
#                     mode=mode,
#                     dirpath=dirpath,
#                     **ckpt_cfg.kwargs,
#                     save_last=False,
#                     save_weights_only=False,
#                     save_on_train_epoch_end=False,
#                     enable_version_counter=False,
#                     auto_insert_metric_name=False,
#                     verbose=False
#                 )
#             )
#         return ckpt_callbacks

#     def setup_early_stopping_callbacks():
#         """Create and return early stopping callbacks."""
#         es_callbacks = []
#         for es_config in config.diagnostics.early_stoppings:
#             OmegaConf.set_readonly(es_config, False)
#             es_config['monitor'] = get_monitored_metric_name(monitored_metrics, es_config.monitor)
#             OmegaConf.set_readonly(es_config, True)

#             es_callbacks.append(
#                 EarlyStoppingRollout(
#                     **es_config,
#                     check_finite=True,
#                     verbose=True,
#                     strict=True,
#                     log_rank_zero_only=True,
#                     timestep=config.data.timestep
#                 )
#             )
#         return es_callbacks

#     def setup_rollout_eval_callbacks():
#         """Create and return rollout evaluation callbacks."""
#         callbacks = []

#         if config.diagnostics.plot.plot_sample:
#             callbacks.append(PlotSample(config, val_dset_len))

#         if config.diagnostics.plot.loss_map:
#             callbacks.append(ForecastingLossMapPlot(config, val_dset_len))

#         if config.diagnostics.plot.loss_bar:
#             callbacks.append(ForecastingLossBarPlot(config, val_dset_len))

#         if config.diagnostics.plot.plot_spectral_loss:
#             callbacks.append(PlotPowerSpectrum(config, val_dset_len))

#         rollout_eval = RolloutEval(
#             config=config,
#             val_dset_len=val_dset_len,
#             callbacks=callbacks
#         )
#         return rollout_eval

#     def setup_convergence_monitoring_callbacks():
#         cm_callbacks = []

#         if any([config.diagnostics.log.wandb.enabled, config.diagnostics.log.mlflow.enabled]):
#             from pytorch_lightning.callbacks import LearningRateMonitor
#             cm_callbacks.append(LearningRateMonitor(logging_interval="step"))
        
#         # TODO - add the experimental observations of gradients as a configurable here (move it to anemoi-experimental)

#         return cm_callbacks

#     def setup_model_averaging_callbacks():
#         ma_callbacks = []
#         if config.training.swa.enabled:
#             from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
#             ma_callbacks.append(
#                 StochasticWeightAveraging(
#                     swa_lrs=config.training.swa.lr,
#                     swa_epoch_start=min(int(0.75 * config.training.max_epochs), config.training.max_epochs - 1),
#                     annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
#                     annealing_strategy="cos",
#                 )
#             )
#         return ma_callbacks

#     # Add all checkpoint-related callbacks
#     trainer_callbacks.extend(setup_checkpoint_callbacks())

#     # Add early stopping callbacks
#     trainer_callbacks.extend(setup_early_stopping_callbacks())

#     # Add rollout evaluation callbacks
#     trainer_callbacks.append(setup_rollout_eval_callbacks())


#     # Add convergence monitoring callbacks
#     trainer_callbacks.extend(setup_convergence_monitoring_callbacks())

#     # Add model averaging callbacks
#     trainer_callbacks.extend(setup_model_averaging_callbacks())
    
#     # Add other miscellaneous callbacks
#     trainer_callbacks.append(MemCleanUpCallback())

#     # Add weight grad output logger callback
#     trainer_callbacks.append(WeightGradOutputLoggerCallback())

#     # Add parent UUID callback
#     trainer_callbacks.append(ParentUUIDCallback(config))

#     if config.diagnostics.plot.learned_features:
#         LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
#         trainer_callbacks.append(GraphTrainableFeaturesPlot(config))

#     return trainer_callbacks
