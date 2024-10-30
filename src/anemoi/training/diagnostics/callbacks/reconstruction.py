import copy

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Optional
from zipfile import ZipFile

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torchinfo
from omegaconf import DictConfig, OmegaConf
from omegaconf import ListConfig
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from anemoi.training.diagnostics.callbacks import BasePlotCallback, GraphTrainableFeaturesPlot, ParentUUIDCallback, WeightGradOutputLoggerCallback
from torch.utils.checkpoint import checkpoint
from anemoi.training.diagnostics.callbacks import BaseLossBarPlot
from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback
from anemoi.training.diagnostics.plots.plots import plot_reconstructed_multilevel_sample
from anemoi.training.diagnostics.plots.plots import plot_loss_map
from anemoi.training.diagnostics.plots.plots import plot_loss
from anemoi.training.diagnostics.plots.plots import plot_power_spectrum
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback

from anemoi.training.diagnostics.callbacks import BasePlotCallback, BaseLossMapPlot
from anemoi.training.diagnostics.callbacks import safe_cast_to_numpy
from anemoi.training.diagnostics.callbacks.common_callbacks import get_common_callbacks
import logging

from anemoi.training.diagnostics.callbacks import get_time_step, increment_to_hours, generate_time_steps

LOGGER = logging.getLogger(__name__)

class ReconstructionLossBarPlot(BaseLossBarPlot):
    """Plots the reconstruction loss accumulated over validation batches and printed once per validation epoch."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.counter = 0
        self.loss_map_accum = defaultdict(lambda: None)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int) -> None:
        if self.op_on_this_batch(batch_idx):
            self.accumulate(trainer, pl_module, outputs, batch)
            self.counter += 1

    def accumulate(self, trainer, pl_module, outputs, batch) -> None:
        LOGGER.debug("Parameters to plot: %s", self.config.diagnostics.plot.parameters)

        x_target = outputs[1]["x_target"]
        x_rec = outputs[1]["x_rec"]
        z_mu = outputs[1]["z_mu"]
        z_logvar = outputs[1]["z_logvar"]

        loss_ = pl_module.loss(
            x_rec,
            x_target,
            z_mu=z_mu,
            z_logvar=z_logvar,
            squash=False,
        ).sum(0) # ( nvar)

        loss_div = loss_[pl_module.loss.reconstruction_loss.name]  # (latlon, nvar)

        if self.loss_map_accum["reconstruction"] is None:
            self.loss_map_accum["reconstruction"] = loss_div
        else:
            self.loss_map_accum["reconstruction"] += loss_div

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.plot(trainer, pl_module)

        keys = list(self.loss_map_accum.keys())
        for k in keys:
            del self.loss_map_accum[k]
        self.loss_map_accum = defaultdict(lambda: None)

        self.counter = 0
        torch.cuda.empty_cache()

    def _plot(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,) -> None:

        parameter_names = list(pl_module.data_indices.model.output.name_to_index.keys())
        paramter_positions = list(pl_module.data_indices.model.output.name_to_index.values())
        
        # reorder parameter_names by position
        self.parameter_names = [parameter_names[i] for i in np.argsort(paramter_positions)]

        sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group

        loss_avg_recon = self.loss_map_accum["reconstruction"] / self.counter
        loss_avg_recon = loss_avg_recon.mean(dim=0)

        # Reduce the loss_avg to global rank 0
        dist.reduce(loss_avg_recon, dst=0, op=dist.ReduceOp.AVG)

        if pl_module.global_rank == 0:
            fig = plot_loss(safe_cast_to_numpy(loss_avg_recon[sort_by_parameter_group], colors, xticks, legend_patches))
            fig.tight_layout()
            
            loss_name_recon = pl_module.loss.reconstruction_loss.name
            self._output_figure(
                trainer,
                fig,
                tag=f"val_loss_barplot_{loss_name_recon}_epoch{epoch:03d}_batch{batch_idx:04d}",
                exp_log_tag=f"val_loss_barplot_{loss_name_recon}",
            )

class ReconstructionLossMapPlot(BaseLossMapPlot):
    def __init__(self, config, val_dset_len, **kwargs) -> None:
        super().__init__(config, val_dset_len, **kwargs)
        self.loss_map_accum = defaultdict(lambda: None)

    def accumulate(self, trainer, pl_module, outputs) -> None:
        x_target = outputs[1]["x_target"]
        x_rec = outputs[1]["x_rec"]
        z_mu = outputs[1]["z_mu"]
        z_logvar = outputs[1]["z_logvar"]

        loss_map = pl_module.loss(
            x_rec,
            x_target,
            z_mu=z_mu,
            z_logvar=z_logvar,
            squash=False
        ) # (latlon, nvar)

        divergence_loss = loss_map[pl_module.loss.divergence_loss.name].sum(dim=-1, keepdim=True)
        self.loss_map_accum["divergence"] = self.loss_map_accum.get("divergence", 0) + divergence_loss

        reconstruction_loss = loss_map[pl_module.loss.reconstruction_loss.name]
        self.loss_map_accum["reconstruction"] = self.loss_map_accum.get("reconstruction", 0) + reconstruction_loss

    def _plot(self, trainer, pl_module) -> None:
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.cpu().numpy())
        
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: name for name in self.config.diagnostics.plot.parameters
        }

        loss_map_avg_recon = self.loss_map_accum["reconstruction"] / self.counter
        loss_map_avg_div = self.loss_map_accum["divergence"] / self.counter

        dist.reduce(loss_map_avg_recon, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(loss_map_avg_div, dst=0, op=dist.ReduceOp.AVG)

        if pl_module.global_rank == 0:
            # Loss map for divergence
            fig, _ = plot_loss_map(
                {0: "latent"},
                safe_cast_to_numpy(np.rad2deg(pl_module.latlons_hidden.clone())),
                safe_cast_to_numpy(loss_map_avg_div),
            )
            fig.tight_layout()
            self._output_figure(trainer, fig, f"val_divergence_lossmap_epoch{trainer.current_epoch:03d}_gstep{trainer.global_step:06d}", "val_map_div")

            # Loss map specifically for map sub - region
            fig, _ = plot_loss_map(
                plot_parameters_dict,
                self.latlons,
                safe_cast_to_numpy(loss_map_avg_recon),
            )
            fig.tight_layout()
            self._output_figure(trainer, fig, f"val_reconstruction_lossmap_epoch{trainer.current_epoch:03d}_gstep{trainer.global_step:06d}", "val_map_recon")

    def reset(self) -> None:
        self.loss_map_accum = defaultdict(lambda: None)

class PlotReconstructedSample(BasePlotCallback):
    """Plots a denormalized reconstructed sample: input and reconstruction."""

    def __init__(self, config, val_dset_len, **kwargs) -> None:
        super().__init__(config, val_dset_len, **kwargs)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def _plot(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        epoch,
    ) -> None:
        # Build dictionary of indices and parameters to be plotted
        plot_parameters_dict = {
            pl_module.data_indices.model.input.name_to_index[name]: (name, True) for name in self.config.diagnostics.plot.parameters
        }

        # When running in Async mode, it might happen that in the last epoch these tensors
        # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
        # but internal ones would be on the cpu), The lines below allow to address this problem
        if self.normalizer is None:
            # Copy to be used across all the training cycle
            self.normalizer = copy.deepcopy(pl_module.model.normalizer).cpu()
            # self.normalizer = pl_module.model.normalizer
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        x_input = outputs[1]["x_inp"].to("cpu")
        x_rec = outputs[1]["x_rec"].to("cpu")

        x_input = x_input[self.sample_idx, 0, ..., pl_module.data_indices.data.output.full]

        x_input = self.normalizer.denormalize(
            x_input,
            in_place=False,
            data_index=pl_module.data_indices.data.output.full,
        ).numpy()

        x_rec = x_rec[self.sample_idx, ...]
        x_rec = torch.mean(x_rec, dim=0)  # -> Reduce the ensemble dimension by taking the mean????

        x_rec = self.normalizer.denormalize(
            x_rec,
            in_place=False,
            data_index=pl_module.data_indices.data.output.full,
        ).numpy()

        fig = plot_reconstructed_multilevel_sample(
            plot_parameters_dict,
            self.latlons,
            x_input.squeeze(),
            x_rec.squeeze(),
        )

        self._output_figure(
            trainer,
            fig,
            tag=f"aifs_vae_val_sample_epoch_{epoch:03d}_batch{batch_idx:04d}",
            exp_log_tag=f"val_pred_sample_rank{local_rank:01d}",
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.op_on_this_batch(
            batch_idx):
            self.plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)

class SpectralAnalysisPlot(BasePlotCallback):
    """Plots spectral comparison of target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

    - Power Spectrum
    """

    def __init__(self, config, val_dset_len, **kwargs) -> None:
        super().__init__(config, val_dset_len, op_on_batch=True, **kwargs)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if SpectralAnalysisPlot.op_on_this_batch(
            batch_idx, pl_module.ens_comm_group_rank, self.plot_frequency, self.min_iter_to_plot
        ):
            self.plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)

    def _plot(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        epoch,
    ) -> None:
        if self.normalizer is None:
            # self.normalizer = copy.deepcopy(pl_module.model.normalizer).cpu()
            self.normalizer = pl_module.model.normalizer

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.to("cpu", non_blocking=True).numpy())

        diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic

        if self.flag_reconstruction:
            plot_parameters_dict_spectrum = {
                pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
                for name in self.eval_plot_parameters
            }

            # This branch for reconstruction training in general
            multi_step = pl_module.multi_step

            _ = batch[0][
                self.sample_idx,
                :multi_step,
                ...,
            ]
            input_ = safe_cast_to_numpy(
                self.normalizer.denormalize(_, in_place=True),
            )

            preds_ = torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...] for x in outputs["preds_denorm"])).to("cpu")

            for step in range(multi_step):
                if self.log_step is True or (
                    (isinstance(self.log_step, list) or isinstance(self.log_step, ListConfig))
                    and step + 1 in self.log_step
                ):
                    x = input_[0, ...].squeeze()
                    target = input_[step, ...].squeeze()
                    y_hat = preds_[step, ...].mean(0)

                    # Calculate based on the mean of the ensembles
                    fig, self.inner_subplots_kwargs = plot_power_spectrum(
                        plot_parameters_dict_spectrum, self.latlons, x, target, y_hat
                    )

                    step_str = get_time_step(self.config.data.timestep, step + 1)

                    self._output_figure(
                        trainer,
                        fig,
                        tag=f"pred_val_spec_s{step_str}_epoch_{epoch:03d}_batch{batch_idx:04d}",
                        exp_log_tag=f"val_pred_spec_step_{step_str}",
                    )

class PlotReconstructionPowerSpectrum(BasePlotCallback):
    """Plots power spectrum metrics comparing target and prediction."""

    def __init__(self, config, val_dset_len, **kwargs):
        """Initialise the PlotPowerSpectrum callback.
        
        # TODO: (Rilwan Adewoyin): Think about what the default case should be for if spectrum should plot tendencies or state for diagnostic variables.

        The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

        Parameters
        ----------
        config : OmegaConf
            Config object
        """
        super().__init__(config, op_on_batch=True, val_dset_len=val_dset_len, **kwargs)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

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
        if self.config.diagnostics.plot.parameters_spectrum is None:
            pass

        logger = trainer.logger

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        
        # NOTE: currently we investigate spatial spectrum only

        
        x_inp_postprocessed = outputs["x_inp_postprocessed"]
        x_target_postprocessed = outputs["x_target_postprocessed"]
        x_rec_postprocessed = outputs["x_rec_postprocessed"]
        multi_step = x_inp_postprocessed.shape[1]
        
        for t in range(multi_step):
            diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic

            plot_parameters_dict_spectrum = {
                pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
                for name in self.config.diagnostics.plot.parameters_spectrum
            }

            fig = plot_power_spectrum(
                plot_parameters_dict_spectrum,
                self.latlons,
                x_inp_postprocessed[self.sample_idx, t, ...].squeeze(),
                x_target_postprocessed[self.sample_idx, t, ...],
                x_rec_postprocessed[self.sample_idx, t, ...],
            )

            step_str = get_time_step(self.config.data.timestep, t + 1)
            self._output_figure(
                logger,
                fig,
                tag=f"power_spectrum_s{step_str}_epoch_{epoch:03d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"power_spectrum_step_{step_str}_rank{pl_module.local_rank:01d}_rank0",
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

class ReconstructEval(Callback):
    def __init__(self, config, val_dset_len, callbacks: list[Callback]) -> None:
        super().__init__()
        self.frequency = config.diagnostics.eval.frequency
        self.eval_frequency = BasePlotCallback.get_eval_frequency(
            self, config.diagnostics.metrics.rollout_eval.frequency, val_dset_len
        )
        
        self.callbacks_validation_batch_end = [
            cb for cb in callbacks 
            if 'on_validation_batch_end' in cb.__class__.__dict__
        ]
        self.callbacks_validation_epoch_end = [
            cb for cb in callbacks 
            if 'on_validation_epoch_end' in cb.__class__.__dict__
        ]

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int
    ) -> None:
        if self.op_on_this_batch(batch_idx):
            with torch.no_grad():
                outputs = self._eval(pl_module, batch[0], batch_idx)
                for cb in self.callbacks_validation_batch_end:
                    cb.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for callback in self.callbacks_validation_epoch_end:
            callback.on_validation_epoch_end(trainer, pl_module)

    def op_on_this_batch(self, batch_idx):
        callbacks_idx_to_run = [
            idx
            for idx, pcallback in enumerate(self.callbacks_validation_batch_end)
            if pcallback.op_on_this_batch(batch_idx)
        ]
        return (((batch_idx + 1) % self.eval_frequency) == 0) or len(callbacks_idx_to_run) > 0
    
    def _eval(self, pl_module: pl.LightningModule, batch: torch.Tensor, batch_idx: int) -> None:
        loss, metrics, outputs = pl_module._step(
            batch, validation_mode=True, batch_idx=batch_idx, 
            lead_time_to_eval=self.lead_time_to_eval
        )
        self._log(pl_module, loss, metrics, batch.shape[0])
        
        return outputs
    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:
        train_loss_name = pl_module.loss.name
        pl_module.log("rval/loss/{pl_module.loss.name}", loss, on_epoch=True, on_step=False, prog_bar=False, logger=pl_module.logger_enabled, batch_size=bs, sync_dist=True, rank_zero_only=True)
        for mname, mvalue in metrics.items():
            pl_module.log(f"rval/{mname}", mvalue, on_epoch=True, on_step=False, prog_bar=False, logger=pl_module.logger_enabled, batch_size=bs, sync_dist=True, rank_zero_only=True)
        

def setup_reconstruction_eval_callbacks(config, val_dset_len):
    """Create and return reconstruction evaluation callbacks."""
    callbacks = []

    if config.diagnostics.plot.loss_map:
        callbacks.append(ReconstructionLossMapPlot(config, val_dset_len))

    if config.diagnostics.plot.loss_bar:
        callbacks.append(ReconstructionLossBarPlot(config, val_dset_len))

    if config.diagnostics.plot.plot_spectral_loss:
        callbacks.append(SpectralAnalysisPlot(config, val_dset_len))

    if config.diagnostics.plot.plot_reconstructed_sample:
        callbacks.append(PlotReconstructedSample(config, val_dset_len))

    reconstruct_eval = ReconstructEval(
        config=config,
        val_dset_len=val_dset_len,
        callbacks=callbacks
    )
    return reconstruct_eval

def get_callbacks(config: DictConfig, monitored_metrics, val_dset_len) -> list:
    trainer_callbacks = get_common_callbacks(config, monitored_metrics)
    
    # Add reconstruction-specific callbacks
    reconstruct_eval = setup_reconstruction_eval_callbacks(config, val_dset_len)
    trainer_callbacks.append(reconstruct_eval)
    
    return trainer_callbacks

# def get_callbacks(config: DictConfig, monitored_metrics, val_dset_len) -> list:
    """Setup callbacks for PyTorch Lightning trainer for reconstruction tasks.

    Parameters
    ----------
    config : DictConfig
        Job configuration
    monitored_metrics : list
        List of monitored metrics to track during training
    val_dset_len : int
        Length of the validation dataset

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
                dirpath = Path(config.hardware.paths.checkpoints) / next(k for k in ("every_n_train_steps", "train_time_interval", "every_n_epochs") if ckpt_cfg.get(k))

            elif ckpt_cfg.type == "performance":
                OmegaConf.set_readonly(ckpt_cfg, False)
                ckpt_cfg.kwargs['monitor'] = get_monitored_metric_name(monitored_metrics, ckpt_cfg['monitor'])
                OmegaConf.set_readonly(ckpt_cfg, True)
                monitor_name = f"perf_{ckpt_cfg.kwargs['monitor'].replace('/', '_')}"
                dirpath = Path(config.hardware.paths.checkpoints) / f"perf_{monitor_name}"
                filename = "epoch={epoch:03d}-step={step:05d}-" + monitor_name + "-{" + ckpt_cfg.kwargs['monitor'] + ":.5f}"

            ckpt_callbacks.append(
                AnemoiCheckpoint(
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
                    verbose=False
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
                EarlyStopping(
                    **es_config,
                    check_finite=True,
                    verbose=True,
                    strict=True,
                    log_rank_zero_only=True
                )
            )
        return es_callbacks

    def setup_reconstruction_eval_callbacks():
        """Create and return reconstruction evaluation callbacks."""
        callbacks = []

        if config.diagnostics.plot.loss_map:
            callbacks.append(ReconstructionLossMapPlot(config, val_dset_len))

        if config.diagnostics.plot.loss_bar:
            callbacks.append(ReconstructionLossBarPlot(config, val_dset_len))

        if config.diagnostics.plot.plot_spectral_loss:
            callbacks.append(SpectralAnalysisPlot(config, val_dset_len))

        if config.diagnostics.plot.plot_reconstructed_sample:
            callbacks.append(PlotReconstructedSample(config, val_dset_len))

        reconstruct_eval = ReconstructEval(
            config=config,
            val_dset_len=val_dset_len,
            callbacks=callbacks
        )
        return reconstruct_eval

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

    # Add reconstruction evaluation callbacks
    reconstruct_eval = setup_reconstruction_eval_callbacks()
    trainer_callbacks.append(reconstruct_eval)

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