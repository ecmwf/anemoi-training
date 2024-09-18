import copy

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
import torch.distributed as dist
import torchinfo
from omegaconf import DictConfig
from omegaconf import ListConfig
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from anemoi.training.diagnostics.callbacks import BasePlotCallback

from anemoi.training.diagnostics.callbacks import LossBarPlot
from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback
from anemoi.training.diagnostics.plots import plot_reconstructed_multilevel_sample
from anemoi.training.diagnostics.plots import plot_loss_map
from anemoi.training.diagnostics.plots import init_plot_settings
from anemoi.training.diagnostics.plots import plot_graph_features
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_loss
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback

from anemoi.training.diagnostics.callbacks import BasePlotCallback, BaseLossMapPlot

def safe_cast_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array, ensuring that the array is of the
    appropriate type."""
    tensor = tensor.to("cpu")

    if tensor.dtype == torch.bfloat16 or tensor.dtype == torch.float32:
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.float16:
        pass

    return tensor.numpy()

class ReconstructionLossBarPlot(LossBarPlot):
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

        loss_div = loss_[pl_module.loss.reconstruction_loss.log_name]  # (latlon, nvar)

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
        pl_module: pl.Lightning_module,
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
            
            loss_log_name_recon = pl_module.loss.reconstruction_loss.log_name
            self._output_figure(
                trainer,
                fig,
                tag=f"val_loss_barplot_{loss_log_name_recon}_epoch{epoch:03d}_batch{batch_idx:04d}",
                exp_log_tag=f"val_loss_barplot_{loss_log_name_recon}",
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

        # if self.loss_map_accum["divergence"] is None:
        #     self.loss_map_accum["divergence"] = loss_map[pl_module.loss.divergence_loss.log_name].sum(dim=-1, keepdim=True)
        # else:
        #     self.loss_map_accum["divergence"] += loss_map[pl_module.loss.divergence_loss.log_name].sum(dim=-1, keepdim=True)

        # if self.loss_map_accum["reconstruction"] is None:
        #     self.loss_map_accum["reconstruction"] = loss_map[pl_module.loss.reconstruction_loss.log_name]
        # else:
        #     self.loss_map_accum["reconstruction"] += loss_map[pl_module.loss.reconstruction_loss.log_name]

        divergence_loss = loss_map[pl_module.loss.divergence_loss.log_name].sum(dim=-1, keepdim=True)
        self.loss_map_accum["divergence"] = self.loss_map_accum.get("divergence", 0) + divergence_loss

        reconstruction_loss = loss_map[pl_module.loss.reconstruction_loss.log_name]
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
        if PredictedEnsemblePlot.op_on_this_batch(
            batch_idx, pl_module.ens_comm_group_rank, self.plot_frequency, self.min_iter_to_plot
        ):
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

        if self.flag_rollout:
            plot_parameters_dict_spectrum = {
                pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
                for name in self.eval_plot_parameters
            }

            # This branch for forecasting training in general
            rollout_steps = len(outputs["preds"])

            _ = batch[0][
                self.sample_idx,
                pl_module.multi_step - 1 : pl_module.multi_step + rollout_steps + 1,
                ...,
            ]
            input_ = safe_cast_to_numpy(
                self.normalizer.denormalize(_, in_place=True),
            )

            preds_ = torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...] for x in outputs["preds_denorm"])).to("cpu")

            rollout_steps = len(preds_)
            for rollout_step in range(rollout_steps):
                if self.log_rstep is True or (
                    (isinstance(self.log_rstep, list) or isinstance(self.log_rstep, ListConfig))
                    and rollout_step + 1 in self.log_rstep
                ):
                    x = input_[0, ...].squeeze()
                    target = input_[rollout_step + 1, ...].squeeze()
                    y_hat = preds_[rollout_step, ...].mean(0)

                    # Calculate based on the mean of the ensembles
                    fig, self.inner_subplots_kwargs = plot_power_spectrum(
                        plot_parameters_dict_spectrum, self.latlons, x, target, y_hat
                    )

                    rollout_step_str = get_time_step(self.config.data.timestep, rollout_step + 1)

                    self._output_figure(
                        trainer,
                        fig,
                        tag=f"pred_val_spec_r{rollout_step_str}_epoch_{epoch:03d}_batch{batch_idx:04d}",
                        exp_log_tag=f"val_pred_spec_rollout_{rollout_step_str}",
                    )

        if self.flag_reconstruction:
            plot_parameters_dict_spectrum = {
                pl_module.data_indices.model.output.name_to_index[name]: (name, False) for name in self.eval_plot_parameters
            }

            x_input = outputs[1]["x_inp"]
            x_rec = outputs[1]["x_rec"]
            x_target = outputs[1]["x_target"]

            x_input = x_input[self.sample_idx, 0, ..., pl_module.data_indices.data.output.full]
            x_input = self.normalizer.denormalize(
                x_input,
                in_place=False,
                data_index=pl_module.data_indices.data.output.full,
            )

            x_rec = x_rec[self.sample_idx, ...]
            # reduce in ensemble dimension by taking mean

            x_rec = torch.mean(x_rec, dim=0)

            x_rec = self.normalizer.denormalize(
                x_rec,
                in_place=False,
                data_index=pl_module.data_indices.data.output.full,
            )

            x_target = x_target[self.sample_idx, ...]
            x_target = self.normalizer.denormalize(
                x_target,
                in_place=False,
                data_index=pl_module.data_indices.data.output.full,
            )

            # TODO: to work with ensembles this will have to change to do mean over ensemble dimension
            x = safe_cast_to_numpy(x_input.squeeze())
            target = safe_cast_to_numpy(x_target.squeeze())
            y_hat = safe_cast_to_numpy(x_rec.squeeze())

            fig, self.inner_subplots_kwargs = plot_power_spectrum(plot_parameters_dict_spectrum, self.latlons, x, target, y_hat)

            self._output_figure(
                trainer,
                fig,
                epoch=pl_module.validation_epoch,
                tag=f"pred_val_spec_epoch_{epoch:03d}_batch{batch_idx:04d}",
                exp_log_tag="val_pred_spec",
            )




def get_callbacks(config: DictConfig, monitored_metrics) -> list:  # noqa: C901
    """Setup callbacks for PyTorch Lightning trainer.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    List
        A list of PyTorch Lightning callbacks

    """
    checkpoint_settings = {
        "dirpath": config.hardware.paths.checkpoints,
        "verbose": False,
        # save weights, optimizer states, LR-schedule states, hyperparameters etc.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#contents-of-a-checkpoint
        "save_weights_only": False,
        "auto_insert_metric_name": False,
        # save after every validation epoch, if we've improved
        "save_on_train_epoch_end": False,
        "enable_version_counter": False,
    }

    ckpt_frequency_save_dict = {}
    for key, frequency_dict in config.diagnostics.checkpoint.items():
        frequency = frequency_dict["save_frequency"]
        n_saved = frequency_dict["num_models_saved"]
        if key == "every_n_minutes" and frequency_dict["save_frequency"] is not None:
            target = "train_time_interval"
            frequency = timedelta(minutes=frequency_dict["save_frequency"])
        else:
            target = key
        ckpt_frequency_save_dict[target] = (config.hardware.files.checkpoint[key], frequency, n_saved)

    trainer_callbacks = []
    if not config.diagnostics.profiler:
        for save_key, (name, save_frequency, save_n_models) in ckpt_frequency_save_dict.items():
            if save_frequency is not None:
                LOGGER.debug("Checkpoint callback at %s = %s ...", save_key, save_frequency)
                trainer_callbacks.extend(
                    # save_top_k: the save_top_k flag can either save the best or the last k checkpoints
                    # depending on the monitor flag on ModelCheckpoint.
                    # See https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html for reference
                    [
                        AnemoiCheckpoint(
                            config=config,
                            filename=name,
                            save_last=True,
                            **{save_key: save_frequency},
                            # if save_top_k == k, last k models saved; if save_top_k == -1, all models are saved
                            save_top_k=save_n_models,
                            monitor="step",
                            mode="max",
                            **checkpoint_settings,
                        ),
                    ],
                )
            else:
                LOGGER.debug("Not setting up a checkpoint callback with %s", save_key)
    else:
        # the tensorboard logger + pytorch profiler cause pickling errors when writing checkpoints
        LOGGER.warning("Profiling is enabled - will not write any training or inference model checkpoints!")

    if any([config.diagnostics.log.wandb.enabled, config.diagnostics.log.mlflow.enabled]):
        from pytorch_lightning.callbacks import LearningRateMonitor

        trainer_callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
                log_momentum=False,
            ),
        )

    if config.diagnostics.eval.enabled:
        trainer_callbacks.append(RolloutEval(config))

    if config.diagnostics.plot.enabled:
        trainer_callbacks.extend(
            [
                PlotLoss(config),
                PlotSample(config),
            ],
        )
        if (config.diagnostics.plot.parameters_histogram or config.diagnostics.plot.parameters_spectrum) is not None:
            trainer_callbacks.extend([PlotAdditionalMetrics(config)])

    if config.training.swa.enabled:
        from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

        trainer_callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.training.swa.lr,
                swa_epoch_start=min(
                    int(0.75 * config.training.max_epochs),
                    config.training.max_epochs - 1,
                ),
                annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
                annealing_strategy="cos",
                device=None,
            ),
        )

    # Early Stopping Callback
    if config.diagnostics.early_stopping.enabled:
        es_monitor = config.diagnostics.early_stopping.monitor

        assert (
            es_monitor == "default" or es_monitor in monitored_metrics
        ), f"""Monitored value:={es_monitor} must either be the loss function,
                    or a stated validation metric!"""

        if es_monitor == "default":
            es_monitor = next((mname for mname in monitored_metrics if mname.startswith("val/loss")), None)
            assert es_monitor is not None, f"Default monitor value not found in monitored metrics: {monitored_metrics}"

        LOGGER.warning(f"Setting up an early stopping callback - monitoring {es_monitor} ...")

        es_cb = EarlyStopping(
            monitor=es_monitor,
            patience=config.diagnostics.early_stopping.patience,
            mode=config.diagnostics.early_stopping.mode,
            check_finite=True,
            verbose=True,
            strict=True,
            log_rank_zero_only=True,
        )

        trainer_callbacks.append(
            es_cb,
        )


    trainer_callbacks.append(ParentUUIDCallback(config))

    if config.diagnostics.plot.learned_features:
        LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
        trainer_callbacks.append(GraphTrainableFeaturesPlot(config))
    return trainer_callbacks