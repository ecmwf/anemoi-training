# (C) 2023 Anemoi, Inc. All rights reserved.

from anemoi.training.diagnostics.plots.plots_ensemble import plot_spread_skill, plot_spread_skill_bins
import copy
import csv
import io
import json
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Optional
from weakref import proxy
from zipfile import ZipFile

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
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

from anemoi.training.diagnostics.metrics import SpreadSkill

from anemoi.training.diagnostics.plots.plots import init_plot_settings
from anemoi.training.diagnostics.plots.plots import plot_graph_features
from anemoi.training.diagnostics.plots.plots_ensemble import plot_histogram
from anemoi.training.diagnostics.plots.plots import plot_loss
from anemoi.training.diagnostics.plots.plots_ensemble import plot_power_spectrum
from anemoi.training.diagnostics.plots.plots import plot_predicted_multilevel_flat_sample
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback

# function should take
from functools import lru_cache
from functools import wraps
from anemoi.training.losses.utils import get_monitored_metric_name
from torch import Tensor
from pytorch_lightning.callbacks import LearningRateMonitor
from anemoi.training.diagnostics.callbacks import (
    GraphTrainableFeaturesPlot,
    BasePlotCallback,
    BaseLossMapPlot,
)
from anemoi.training.diagnostics.callbacks.forecast import (
    EarlyStoppingRollout,
    ForecastingLossBarPlot,
    ForecastingLossMapPlot,
    AnemoiCheckpointRollout,
    PlotPowerSpectrum,
    PlotSample,
)

from anemoi.training.diagnostics import safe_cast_to_numpy, get_time_step

from anemoi.training.diagnostics.plots.plots_ensemble import plot_predicted_ensemble
from anemoi.training.diagnostics.plots.plots_ensemble import plot_rank_histograms
from anemoi.training.diagnostics.plots.plots_ensemble import plot_spread_skill
from anemoi.training.diagnostics.plots.plots_ensemble import plot_spread_skill_bins

from anemoi.training.diagnostics.callbacks import WeightGradOutputLoggerCallback
import logging
from anemoi.models.distributed.graph import gather_tensor

LOGGER = logging.getLogger(__name__)


class RolloutEvalEns(Callback):
    def __init__(
        self, config, val_dset_len, callbacks_validation_batch_end: list, 
        callbacks_validation_epoch_end: list
    ) -> None:
        super().__init__()
        self.rollout = config.diagnostics.metrics.rollout_eval.rollout
        self.frequency = config.diagnostics.eval.frequency
        self.eval_frequency = BasePlotCallback.get_eval_frequency(
            self, config.diagnostics.metrics.rollout_eval.frequency, val_dset_len
        )
        self.lead_time_to_eval = config.diagnostics.eval.lead_time_to_eval
        self.callbacks_validation_batch_end = callbacks_validation_batch_end
        self.callbacks_validation_epoch_end = callbacks_validation_epoch_end

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int
    ) -> None:
        if self.op_on_this_batch(batch_idx):
            precision_mapping = {"16-mixed": torch.float16, "bf16-mixed": torch.bfloat16}
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = (
                torch.autocast(device_type=batch.device.type, dtype=dtype)
                if dtype is not None
                else nullcontext()
            )

            with context:
                with torch.no_grad():
                    outputs = self._eval(pl_module, batch[0], batch_idx)
                    for cb in self.callbacks_validation_batch_end:
                        cb.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for callback in self.callbacks_validation_epoch_end:
            callback.on_validation_epoch_end(trainer, pl_module)

    def _eval(self, pl_module: pl.LightningModule, batch: torch.Tensor, batch_idx: int) -> None:
        loss, metrics, outputs = pl_module._step(
            batch, validation_mode=True, batch_idx=batch_idx, 
            lead_time_to_eval=self.lead_time_to_eval
        )
        self._log(pl_module, loss, metrics, batch.shape[0])
        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + self.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ]
        data = safe_cast_to_numpy(self.post_processors_state_state(input_tensor))
        outputs["data"] = data
        return outputs

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:
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

    def op_on_this_batch(self, batch_idx):
        callbacks_idx_to_run = [
            idx
            for idx, pcallback in enumerate(self.callbacks_validation_batch_end)
            if pcallback.op_on_this_batch(batch_idx)
        ]
        return (((batch_idx + 1) % self.eval_frequency) == 0) or len(callbacks_idx_to_run) > 0


class SpreadSkillPlot(BasePlotCallback):
    def __init__(self, config, val_dset_len, **kwargs):
        super().__init__(config, op_on_batch=True, val_dset_len=val_dset_len, **kwargs)

        self.spread_skill = SpreadSkill(
            rollout=config.diagnostics.metrics.rollout_eval.rollout,
            nvar=len(config.diagnostics.plot.parameters),
            nbins=config.diagnostics.metrics.rollout_eval.num_bins,
            time_step=int(config.data.timestep[:-1]),
        )

        self.lead_time_to_eval = self.config.diagnostics.eval.lead_time_to_eval

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int,
    ) -> None:
        # TODO(rilwan-ade): add pl_module.ens_comm_group_rank logic so only one ensemble rank plots
        if self.op_on_this_batch(batch_idx):
            node_weights = pl_module.graph_data[("era", "to", "era")].node_weights.to(device=pl_module.device)

            plot_parameters_dict = {
                pl_module.data_indices.model.output.name_to_index[name]: name
                for name in self.eval_plot_parameters
                if name in pl_module.data_indices.model.output.name_to_index
            }

            preds_denorm = outputs["preds_denorm"]
            targets_denorm = outputs["targets_denorm"]

            rollout_steps = len(preds_denorm)

            rmse = torch.zeros(
                (rollout_steps, len(self.eval_plot_parameters)),
                dtype=batch[0].dtype,
                device=pl_module.device,
            )
            spread = torch.zeros_like(rmse)
            binned_rmse = torch.zeros(
                (rollout_steps, len(self.eval_plot_parameters), self.spread_skill.nbins - 1),
                dtype=batch[0].dtype,
                device=pl_module.device,
            )
            binned_spread = torch.zeros_like(binned_rmse)

            # for rollout_step in range(rollout_steps):
            for idx in range(len(outputs["y_pred"])):
                rollout_step = self.lead_time_to_eval[idx]
                pred_denorm = preds_denorm[rollout_step]
                target_denorm = targets_denorm[rollout_step]

                for midx, (pidx, _) in enumerate(plot_parameters_dict.items()):
                    (
                        rmse[idx, midx],
                        spread[idx, midx],
                        binned_rmse[idx, midx],
                        binned_spread[idx, midx],
                    ) = self.spread_skill.calculate_spread_skill(pred_denorm, target_denorm, pidx, node_weights)

            _ = self.spread_skill(rmse, spread, binned_rmse, binned_spread, pl_module.device)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: name for name in self.config.diagnostics.plot.parameters
        }
        if self.spread_skill.num_updates != 0:
            rmse, spread, bins_rmse, bins_spread = (r for r in self.spread_skill.compute())
            rmse, spread, bins_rmse, bins_spread = (
                safe_cast_to_numpy(rmse),
                safe_cast_to_numpy(spread),
                safe_cast_to_numpy(bins_rmse),
                safe_cast_to_numpy(bins_spread),
            )
            fig = plot_spread_skill(plot_parameters_dict, (rmse, spread), self.spread_skill.time_step, self.lead_time_to_eval)
            self._output_figure(
                trainer,
                fig,
                epoch=trainer.current_epoch,
                tag="ens_spread_skill",
                exp_log_tag=f"val_spread_skill_{pl_module.global_rank}",
            )
            fig = plot_spread_skill_bins(plot_parameters_dict, (bins_rmse, bins_spread), self.spread_skill.time_step)


            fig.tight_layout()


            self._output_figure(
                trainer,
                fig,
                epoch=trainer.current_epoch,
                tag="ens_spread_skill_bins",
                exp_log_tag=f"val_spread_skill_bins_{pl_module.global_rank}",
            )

class RankHistogramPlot(BasePlotCallback):
    def __init__(self, config, ranks):
        super().__init__(config)
        self.ranks = ranks

    def _plot(self, trainer, pl_module, epoch_tag) -> None:
        diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
        plot_parameters_dict = {pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics) for name in self.config.diagnostics.plot.parameters}
        fig = plot_rank_histograms(plot_parameters_dict, self.ranks.ranks.cpu().numpy())
        self._output_figure(trainer, fig, epoch=epoch_tag, tag="ens_rank_hist", exp_log_tag=f"val_rank_hist_{pl_module.global_rank}")
        self.ranks.reset()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.ens_comm_group_rank == 0:
            self.plot(trainer.logger, pl_module, epoch_tag=trainer.current_epoch)

class PlotEnsembleInitialConditions(BasePlotCallback):
    def __init__(self, config):
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def _plot(self, trainer, pl_module, ens_ic, batch_idx, epoch) -> None:
        diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
        plot_parameters_dict = {pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics) for name in self.config.diagnostics.plot.parameters}
        if self.post_processors_state is None:
            self.post_processors_state = copy.deepcopy(pl_module.model.post_processors_state).cpu()
        input_tensor = ens_ic.cpu()
        denorm_ens_ic = self.post_processors_state.processors.normalizer.inverse_transform(input_tensor, in_place=False, data_index=pl_module.data_indices.data.input.full).numpy()
        for step in range(pl_module.multi_step):
            fig = plot_predicted_ensemble(plot_parameters_dict, 1, np.rad2deg(pl_module.latlons_data.numpy()), self.config.diagnostics.plot.accumulation_levels_plot, self.config.diagnostics.plot.cmap_accumulation, denorm_ens_ic[self.sample_idx, step, ...].squeeze(), denorm_ens_ic[self.sample_idx, step, ...].squeeze(), scatter=self.scatter_plotting, initial_condition=True)
            fig.tight_layout()
            self._output_figure(trainer, fig, epoch=epoch, tag=f"ens_ic_val_mstep{step:02d}_batch{batch_idx:05d}_rank{pl_module.global_rank:03d}", exp_log_tag=f"ens_ic_val_mstep{step:02d}_rank{pl_module.global_rank:03d}")

    def _gather_group_initial_conditions(self, pl_module: pl.LightningModule, my_ens_ic: torch.Tensor) -> torch.Tensor:
        group_ens_ic = gather_tensor(my_ens_ic, dim=1, shapes=[my_ens_ic.shape] * pl_module.ens_comm_group_size, mgroup=pl_module.ens_comm_group)
        return group_ens_ic

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int) -> None:
        
        group_ens_ic = self._gather_group_initial_conditions(pl_module, outputs[-1])
        group_ens_ic = einops.rearrange(group_ens_ic, "bs s e latlon v -> bs s v latlon e")
        group_ens_ic = group_ens_ic @ pl_module._gather_matrix
        group_ens_ic = einops.rearrange(group_ens_ic, "bs s v latlon e -> bs s e latlon v")
        if self.op_on_this_batch(batch_idx) and pl_module.ens_comm_group_rank == 0:
            self.plot(trainer.logger, pl_module, group_ens_ic, batch_idx, epoch=trainer.current_epoch)

class PlotEnsSample(PlotSample):
    def _generate_plot_fn(self, plot_parameters_dict, plot_per_sample, latlons, accum_levels, cmap_accum, x, y_true, y_pred, scatter):
        fig = plot_predicted_ensemble(plot_parameters_dict, 4, latlons, accum_levels, cmap_accum, y_true, y_pred, scatter, initial_condition=False)
        return fig

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int) -> None:
        if self.op_on_this_batch(batch_idx) and pl_module.ens_comm_group_rank == 0:
            self.plot(trainer, pl_module, self._generate_plot_fn, outputs, batch[0], batch_idx, epoch=trainer.current_epoch)

class PlotEnsPowerSpectrum(PlotPowerSpectrum):
    def __init__(self, config, val_dset_len, **kwargs):
        super().__init__(config, op_on_batch=True, val_dset_len=val_dset_len, **kwargs)
        self.ens_indexes_to_plot : list[int] = self.config.diagnostics.plot.ens_idx 

    def _plot_spectrum(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: list, batch: torch.Tensor, batch_idx: int, epoch: int) -> None:

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        
        data = outputs["data"]
        output_tensor = outputs['y_pred_postprocessed']

        for idx in range(len(outputs["y_pred"])):
            rollout_step = self.lead_time_to_eval[idx]

            if self.config.diagnostics.plot.parameters_spectrum is not None:
                diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
                plot_parameters_dict_spectrum = {pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics) for name in self.config.diagnostics.plot.parameters_spectrum}
                

                fig = plot_power_spectrum(plot_parameters_dict_spectrum, self.latlons, data[0, ...].squeeze(), data[rollout_step + 1, ...].squeeze(), safe_cast_to_numpy(output_tensor[idx]))
                
                self._output_figure(trainer.logger, fig, tag=f"gnn_pred_val_spec_rstep_{rollout_step:02d}_epoch_{epoch:03d}_batch{batch_idx:04d}_rank0", exp_log_tag=f"val_pred_spec_rstep_{rollout_step:02d}_rank{pl_module.local_rank:01d}")


def get_ens_callbacks(config: DictConfig, monitored_metrics, val_dset_len) -> list:
    """Setup callbacks for PyTorch Lightning trainer with ensemble-specific modifications.

    Parameters
    ----------
    config : DictConfig
        Configuration object.
    monitored_metrics : list
        List of monitored metrics to track during training
    val_dset_len : int
        Length of the validation dataset.

    Returns
    -------
    list
        A list of PyTorch Lightning callbacks
    """
    trainer_callbacks = []

    def setup_checkpoint_callbacks():
        """Create and return checkpoint-related callbacks."""
        ckpt_callbacks = []
        if config.diagnostics.profiler:
            LOGGER.warning("Profiling is enabled - no checkpoints will be written!")
            return []

        checkpoint_configs = config.diagnostics.checkpoints

        for ckpt_cfg in checkpoint_configs:
            filename, mode, dirpath = None, "max", None

            if ckpt_cfg.type == 'interval':
                dirpath = os.path.join(
                    config.hardware.paths.checkpoints,
                    next(k for k in ("every_n_train_steps", "train_time_interval", "every_n_epochs") if ckpt_cfg.get(k)),
                )

            elif ckpt_cfg.type == "performance":
                OmegaConf.set_readonly(ckpt_cfg, False)
                ckpt_cfg.kwargs['monitor'] = get_monitored_metric_name(monitored_metrics, ckpt_cfg['monitor'])
                OmegaConf.set_readonly(ckpt_cfg, True)
                monitor_name = f"perf_{ckpt_cfg.kwargs['monitor'].replace('/', '_')}"
                dirpath = os.path.join(config.hardware.paths.checkpoints, f"perf_{monitor_name}")
                filename = f"epoch={{epoch}}-step={{step}}-{monitor_name}-{{{ckpt_cfg.kwargs['monitor']}}:.5f}"

            ckpt_callbacks.append(
                AnemoiCheckpointRollout(
                    config=config,
                    filename=filename,
                    mode=mode,
                    dirpath=dirpath,
                    **ckpt_cfg.kwargs,
                    save_last=False,
                    save_weights_only=False,
                    save_on_train_epoch_end=False,
                    enable_version_counter=False,
                    auto_insert_metric_name=False,
                    verbose=False,
                )
            )
        return ckpt_callbacks

    def setup_early_stopping_callbacks():
        """Create and return early stopping callbacks."""
        es_callbacks = []
        for es_config in config.diagnostics.early_stoppings:
            OmegaConf.set_readonly(es_config, False)
            es_config['monitor'] = get_monitored_metric_name(monitored_metrics, es_config.monitor)
            OmegaConf.set_readonly(es_config, True)

            es_callbacks.append(
                EarlyStoppingRollout(
                    **es_config,
                    check_finite=True,
                    verbose=True,
                    strict=True,
                    log_rank_zero_only=True,
                    timestep=config.data.timestep,
                )
            )
        return es_callbacks

    def setup_rollout_eval_callbacks():
        """Create and return rollout evaluation callbacks."""
        validation_batch_end_callbacks = []
        validation_epoch_end_callbacks = []

        # TODO (rilwan-ade): think of better way to implement this??
        if config.diagnostics.plot.loss_map:
            loss_map_plot = ForecastingLossMapPlot(config, val_dset_len)
            validation_batch_end_callbacks.append(loss_map_plot)
            validation_epoch_end_callbacks.append(loss_map_plot)

        if config.diagnostics.plot.loss_bar:
            loss_bar_plot = ForecastingLossBarPlot(config, val_dset_len)
            validation_batch_end_callbacks.append(loss_bar_plot)
            validation_epoch_end_callbacks.append(loss_bar_plot)

        if config.diagnostics.plot.plot_spectral_loss:
            validation_batch_end_callbacks.append(PlotEnsPowerSpectrum(config, val_dset_len))

        # Add ensemble-specific callbacks
        if config.diagnostics.plot.ens_sample:
            validation_batch_end_callbacks.append(PlotEnsSample(config))

        if config.diagnostics.plot.rank_histogram:
            validation_epoch_end_callbacks.append(RankHistogramPlot(config, ranks=None))  # You'll need to pass the appropriate ranks object

        if config.diagnostics.plot.spread_skill_plot:
            validation_batch_end_callbacks.append(SpreadSkillPlot(config, val_dset_len))

        if config.diagnostics.plot.ensemble_initial_conditions:
            validation_batch_end_callbacks.append(PlotEnsembleInitialConditions(config))

        rollout_eval = RolloutEvalEns(
            config=config,
            val_dset_len=val_dset_len,
            callbacks_validation_batch_end=validation_batch_end_callbacks,
            callbacks_validation_epoch_end=validation_epoch_end_callbacks,
        )
        return rollout_eval

    def setup_convergence_monitoring_callbacks():
        cm_callbacks = []

        if any([config.diagnostics.log.wandb.enabled, config.diagnostics.log.mlflow.enabled]):
            from pytorch_lightning.callbacks import LearningRateMonitor
            cm_callbacks.append(LearningRateMonitor(logging_interval="step"))

        return cm_callbacks

    def setup_model_averaging_callbacks():
        ma_callbacks = []
        if config.training.swa.enabled:
            from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
            ma_callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=config.training.swa.lr,
                    swa_epoch_start=min(int(0.75 * config.training.max_epochs), config.training.max_epochs - 1),
                    annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
                    annealing_strategy="cos",
                )
            )
        return ma_callbacks

    # Add all checkpoint-related callbacks
    trainer_callbacks.extend(setup_checkpoint_callbacks())

    # Add early stopping callbacks
    trainer_callbacks.extend(setup_early_stopping_callbacks())

    # Add rollout evaluation callbacks
    trainer_callbacks.append(setup_rollout_eval_callbacks())

    # Add convergence monitoring callbacks
    trainer_callbacks.extend(setup_convergence_monitoring_callbacks())

    # Add model averaging callbacks
    trainer_callbacks.extend(setup_model_averaging_callbacks())

    # Add other miscellaneous callbacks
    trainer_callbacks.append(MemCleanUpCallback())

    # Add weight grad output logger callback
    trainer_callbacks.append(WeightGradOutputLoggerCallback())

    # Add parent UUID callback
    trainer_callbacks.append(ParentUUIDCallback(config))

    if config.diagnostics.plot.learned_features:
        LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
        trainer_callbacks.append(GraphTrainableFeaturesPlot(config))

    return trainer_callbacks
