# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# * [WHY ARE CALLBACKS UNDER __init__.py?]
# * This functionality will be restructured in the near future
# * so for now callbacks are under __init__.py

# TODO: (Rilwan-Ade) Make sure that in logs we include epoch and step within epoch to circumvent having to figure out validation epoch
from __future__ import annotations

import logging
import sys
import time
import traceback
import uuid
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torchinfo
from anemoi.utils.checkpoints import save_metadata
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from anemoi.training.diagnostics.plots.plots import init_plot_settings
from anemoi.training.diagnostics.plots.plots import plot_graph_features
from anemoi.training.diagnostics.plots.plots import plot_loss

from typing import TYPE_CHECKING
import cv2
from functools import lru_cache
from functools import wraps

import cartopy.crs as ccrs

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


def tensor_lru_cache(maxsize=128, typed=False):
    def decorator(func):
        cache = lru_cache(maxsize=maxsize, typed=typed)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert Tensor arguments to a hashable type
            key = tuple(arg.tolist() if isinstance(arg, Tensor) else arg for arg in args)
            key_kwargs = {k: (v.tolist() if isinstance(v, Tensor) else v) for k, v in kwargs.items()}
            return cache(func)(*key, **key_kwargs)

        return wrapper

    return decorator


def safe_cast_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array, ensuring that the array is of the
    appropriate type.
    """
    tensor = tensor.to("cpu")

    if tensor.dtype == torch.bfloat16 or tensor.dtype == torch.float32:
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.float16:
        pass

    return tensor.numpy()


@tensor_lru_cache()
def get_time_step(increment: str, step: Tensor) -> str:
    """Return the time step name for a given step number based on increment."""
    inc_hours = increment_to_hours(increment)

    total_hours = step * inc_hours
    days = total_hours / 24

    return f"{days:.2f}d"


@lru_cache
def increment_to_hours(increment: str):
    """Convert time increment string to hours."""
    if increment.endswith("h"):
        return int(increment[:-1])
    if increment.endswith("d"):
        return int(increment[:-1]) * 24
    msg = "Invalid time increment format. Use 'h' for hours and 'd' for days."
    raise ValueError(msg)


@tensor_lru_cache()
def generate_time_steps(increment: str, steps: int):
    """Generate named time steps based on increment and number of steps."""
    inc_hours = increment_to_hours(increment)

    time_steps = []

    for step in range(1, steps + 1):
        total_hours = step * inc_hours
        days = total_hours / 24

        step_name = f"{int(days)}d" if days.is_integer() else f"{days:.2f}d"

        time_steps.append(step_name)

    return time_steps


class ParallelExecutor(ThreadPoolExecutor):
    """Wraps parallel execution and provides accurate information about errors.

    Extends ThreadPoolExecutor to preserve the original traceback and line number.

    Reference: https://stackoverflow.com/questions/19309514/getting-original-line-number-for-exception-in-concurrent-futures/24457608#24457608
    """

    def submit(self, fn: Any, *args, **kwargs) -> Callable:
        """Submits the wrapped function instead of `fn`."""
        return super().submit(self._function_wrapper, fn, *args, **kwargs)

    def _function_wrapper(self, fn: Any, *args: list, **kwargs: dict) -> Callable:
        """Wraps `fn` in order to preserve the traceback of any kind of."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise sys.exc_info()[0](traceback.format_exc()) from exc


class BasePlotCallback(Callback, ABC):
    """Factory for creating a callback that plots data to Experiment Logging."""

    def __init__(self, config: OmegaConf, val_dset_len: int | None = None, op_on: str = "batch") -> None:
        """Initialise the BasePlotCallback abstract base class.

        Parameters
        ----------
        config : OmegaConf
            Config object
        val_dset_len : Optional[int]
            Length of validation dataset
        op_on : str
            Operation on 'batch' or 'epoch'

        """
        super().__init__()
        self.config = config
        self.save_basedir = config.hardware.paths.plots
        self.plot_frequency = self.get_plot_frequency(config.diagnostics.plot.frequency, val_dset_len)
        self.post_processors_state = None
        self.pre_processors_state = None
        self.latlons = None
        init_plot_settings()

        self.plot = self._plot
        self._executor = None

        assert op_on in ["batch", "epoch"], f"Operation on {op_on} not supported"
        self.op_on = op_on

        if self.config.diagnostics.plot.asynchronous:
            self._executor = ParallelExecutor(max_workers=1)
            self._error: BaseException | None = None
            self.plot = self._async_plot

    @rank_zero_only
    def _output_figure(
        self,
        logger: pl.loggers.base.LightningLoggerBase,
        fig: plt.Figure,
        tag: str = "gnn",
        exp_log_tag: str = "val_pred_sample",
    ) -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}.png",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            if self.config.diagnostics.log.wandb.enabled:
                import wandb

                logger.experiment.log({exp_log_tag: wandb.Image(fig)})

            if self.config.diagnostics.log.mlflow.enabled:
                run_id = logger.run_id
                logger.experiment.log_artifact(run_id, str(save_path))

        plt.close(fig)  # cleanup

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Method is called to close the threads."""
        del trainer, pl_module, stage  # unused
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    @abstractmethod
    @rank_zero_only
    def _plot(
        *args: list,
        **kwargs: dict,
    ) -> None: ...

    @rank_zero_only
    def _async_plot(
        self,
        trainer: pl.Trainer,
        *args: list,
        **kwargs: dict,
    ) -> None:
        """To execute the plot function but ensuring we catch any errors."""
        future = self._executor.submit(
            self._plot,
            trainer,
            *args,
            **kwargs,
        )
        # otherwise the error won't be thrown till the validation epoch is finished
        try:
            future.result()
        except Exception:
            LOGGER.exception("Critical error occurred in asynchronous plots.")
            sys.exit(1)

    def get_plot_frequency(self, plot_frequency: float | int, val_dset_len: int | None) -> int:
        self.plot_frequency = None

        # If not operating on batch, just return
        if self.op_on == "epoch":
            assert isinstance(plot_frequency, int), "Plot frequency must be an integer when operating on epoch"
            freq = plot_frequency

        elif self.op_on == "batch":

            if isinstance(plot_frequency, int):
                self.plot_frequency = plot_frequency

            # If frequency is a float, calculate based on dataset length
            elif isinstance(plot_frequency, float):
                freq_rate = plot_frequency

                # Ensure validation dataset length is available
                if val_dset_len is not None:
                    freq = int(val_dset_len * freq_rate)
                else:
                    LOGGER.error(f"Plot frequency could not be determined for {self.__class__.__name__}")
                    msg = f"Plot frequency could not be determined for {self.__class__.__name__}"
                    raise ValueError(msg)

                # Ensure frequency is not lower than the minimum required
                if freq < self.min_iter_to_plot:
                    LOGGER.warning(f"Current plot frequency {freq} too low, setting to {self.min_iter_to_plot}")
                    freq = max(self.min_iter_to_plot, freq)

        return freq

    def op_on_this_batch(self, batch_idx):
        return ((batch_idx + 1) % self.plot_frequency) == 0


class GraphTrainableFeaturesPlot(BasePlotCallback):
    """Visualize the trainable features defined at the data and hidden graph nodes.

    TODO: How best to visualize the learned edge embeddings? Offline, perhaps - using code from @Simon's notebook?

    #TODO: is this to be included in further code???
    """

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the GraphTrainableFeaturesPlot callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__(config)
        self._graph_name_data = config.graph.data
        self._graph_name_hidden = config.graph.hidden

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        latlons: np.ndarray,
        features: np.ndarray,
        epoch: int,
        tag: str,
        exp_log_tag: str,
    ) -> None:
        fig = plot_graph_features(latlons, features)
        self._output_figure(trainer.logger, fig, epoch=epoch, tag=tag, exp_log_tag=exp_log_tag)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model
        graph = pl_module.graph_data.cpu().detach()
        epoch = trainer.current_epoch

        if model.trainable_data is not None:
            data_coords = np.rad2deg(graph[self._graph_name_data, "to", self._graph_name_data].ecoords_rad.numpy())

            self.plot(
                trainer,
                data_coords,
                model.trainable_data.trainable.cpu().detach().numpy(),
                epoch=epoch,
                tag="trainable_data",
                exp_log_tag="trainable_data",
            )

        if model.trainable_hidden is not None:
            hidden_coords = np.rad2deg(
                graph[self._graph_name_hidden, "to", self._graph_name_hidden].hcoords_rad.numpy(),
            )

            self.plot(
                trainer,
                hidden_coords,
                model.trainable_hidden.trainable.cpu().detach().numpy(),
                epoch=epoch,
                tag="trainable_hidden",
                exp_log_tag="trainable_hidden",
            )


class BaseLossBarPlot(BasePlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(self, config: OmegaConf, **kwargs) -> None:
        """Initialise the LossBarPlot callback.

        Parameters
        ----------
        config : OmegaConf
            Object with configuration settings

        """
        super().__init__(config, **kwargs)
        self.parameter_names = None
        self.parameter_groups = self.config.diagnostics.plot.parameter_groups
        if self.parameter_groups is None:
            self.parameter_groups = {}

    @cached_property
    def sort_and_color_by_parameter_group(self) -> tuple[np.ndarray, np.ndarray, dict, list]:
        """Sort parameters by group and prepare colors."""

        def automatically_determine_group(name: str) -> str:
            # first prefix of parameter name is group name
            parts = name.split("_")
            return parts[0]

        # group parameters by their determined group name for > 15 parameters
        if len(self.parameter_names) <= 15:
            # for <= 15 parameters, keep the full name of parameters
            parameters_to_groups = np.array(self.parameter_names)
            sort_by_parameter_group = np.arange(len(self.parameter_names), dtype=int)
        else:
            parameters_to_groups = np.array(
                [
                    next(
                        (
                            group_name
                            for group_name, group_parameters in self.parameter_groups.items()
                            if name in group_parameters
                        ),
                        automatically_determine_group(name),
                    )
                    for name in self.parameter_names
                ],
            )

            unique_group_list, group_inverse, group_counts = np.unique(
                parameters_to_groups,
                return_inverse=True,
                return_counts=True,
            )

            # join parameter groups that appear only once and are not given in config-file
            unique_group_list = np.array(
                [
                    unique_group_list[tn] if count > 1 or unique_group_list[tn] in self.parameter_groups else "other"
                    for tn, count in enumerate(group_counts)
                ],
            )
            parameters_to_groups = unique_group_list[group_inverse]
            unique_group_list, group_inverse = np.unique(parameters_to_groups, return_inverse=True)

            # sort paramters by groups
            sort_by_parameter_group = np.argsort(group_inverse, kind="stable")

        # apply new order to paramters
        sorted_parameter_names = np.array(self.parameter_names)[sort_by_parameter_group]
        parameters_to_groups = parameters_to_groups[sort_by_parameter_group]
        unique_group_list, group_inverse, group_counts = np.unique(
            parameters_to_groups,
            return_inverse=True,
            return_counts=True,
        )

        LOGGER.info("Order of parameters in loss histogram: %s", sorted_parameter_names)

        # get a color per group and project to parameter list
        cmap = "tab10" if len(unique_group_list) <= 10 else "tab20"
        if len(unique_group_list) > 20:
            LOGGER.warning("More than 20 groups detected, but colormap has only 20 colors.")
        # if all groups have count 1 use black color
        bar_color_per_group = (
            "k" if not np.any(group_counts - 1) else plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_group_list)))
        )

        # set x-ticks
        x_tick_positions = np.cumsum(group_counts) - group_counts / 2 - 0.5
        xticks = dict(zip(unique_group_list, x_tick_positions))

        legend_patches = []
        for group_idx, group in enumerate(unique_group_list):
            text_label = f"{group}: "
            string_length = len(text_label)
            for ii in np.where(group_inverse == group_idx)[0]:
                text_label += sorted_parameter_names[ii] + ", "
                string_length += len(sorted_parameter_names[ii]) + 2
                if string_length > 50:
                    # linebreak after 50 characters
                    text_label += "\n"
                    string_length = 0
            legend_patches.append(mpatches.Patch(color=bar_color_per_group[group_idx], label=text_label[:-2]))

        return sort_by_parameter_group, bar_color_per_group[group_inverse], xticks, legend_patches

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.Lightning_module,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        del batch_idx  # unused
        logger = trainer.logger

        parameter_names = list(pl_module.data_indices.model.output.name_to_index.keys())
        paramter_positions = list(pl_module.data_indices.model.output.name_to_index.values())
        # reorder parameter_names by position
        self.parameter_names = [parameter_names[i] for i in np.argsort(paramter_positions)]

        for rollout_step in range(pl_module.rollout):
            y_hat = outputs[1][rollout_step]
            y_true = batch[:, pl_module.multi_step + rollout_step, ..., pl_module.data_indices.data.output.full]
            loss = pl_module.loss(y_hat, y_true, squash=False).mean(0).cpu().numpy()

            sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group
            fig = plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
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


# TODO: (rilwan-ade) make sure that this is added for implementations of forecaster, forecast_ensemble,
class WeightGradOutputLoggerCallback(Callback):
    """Tensorboard Callback."""

    from torch.utils.tensorboard import SummaryWriter

    def __init__(
        self,
        summary_writer: SummaryWriter,
        log_weights: bool,
        log_weights_interval: str,
        log_weights_freq: int | None,
        log_gradients: bool,
        log_gradients_freq: int | None,
        log_clipped_gradients: bool,
        log_clipped_gradients_freq: int,
        log_preds: bool,
        log_preds_freq: int | None,
        log_postproc_preds: int | None,
        log_postproc_preds_freq: int,
    ):
        super().__init__()

        log_periods = ["batch", "epoch"]

        if log_weights is True:
            assert (
                log_weights_freq is None or log_weights_freq > 0
            ), "log_weights_freq must be greater than 0 if log_weights is True"
            assert log_weights_interval in log_periods, "log_weights_interval must be 'batch' or 'epoch'"

        if log_gradients is True:
            assert (
                log_gradients_freq is not None and log_gradients_freq > 0
            ), "log_gradients_freq must be greater than 0 if log_gradients is True"

        if log_clipped_gradients is True:
            assert (
                log_clipped_gradients_freq is not None and log_clipped_gradients_freq > 0
            ), "log_clipped_gradients_freq must be greater than 0 if log_clipped_gradients is True"

        if log_preds is True:
            assert log_preds_freq is not None and log_preds_freq > 0, "log_preds_freq must be greater than 0 if log_preds is True"

        if log_postproc_preds is True:
            assert (
                log_postproc_preds_freq is not None and log_postproc_preds_freq > 0
            ), "log_postproc_preds_freq must be greater than 0 if log_postproc_preds is True"

        self.writer = summary_writer
        self._log_weights = log_weights
        self._log_weights_interval = log_weights_interval
        self._log_weights_freq = log_weights_freq

        self._log_gradients = log_gradients
        self._log_gradients_freq = log_gradients_freq

        self._log_clipped_gradients = log_clipped_gradients
        self._log_clipped_gradients_freq = log_clipped_gradients_freq

        self._log_preds = log_preds
        self._log_preds_freq = log_preds_freq

        self._log_postproc_preds = log_postproc_preds
        self._log_postproc_preds_freq = log_postproc_preds_freq

    def log_parameters(self, named_params: list, idx: int, log_weights=False, log_grads=False, log_clipped_grads=False) -> None:
        for name, param in named_params:
            if log_weights:
                self.writer.add_histogram(f"{name}_weights", param, idx)
            if log_grads and param.grad is not None:
                self.writer.add_histogram(f"{name}_grads", param.grad, idx)
            if log_clipped_grads and param.grad is not None:
                self.writer.add_histogram(f"{name}_clipped_grads", param.grad, idx)

    def log_preds(self, pl_module, outputs: list, idx: int, flag_outputs=False, flag_postproc_outputs=False) -> None:
        if not flag_outputs and not flag_postproc_outputs:
            return

        preds = torch.stack(outputs["y_preds"], dim=1)  # outputs shape: (bs, ts, ens, latlon, nvar )

        preds_denorm = None
        if flag_postproc_outputs:
            preds_denorm = pl_module.model.post_processors(preds, in_place=False)  # ens

        idx_to_name = {
            pl_module.data_indices.model.output.name_to_index[name]: name
            for name in pl_module.data_indices.model.output.name_to_index
        }

        for i in range(preds.shape[-1]):
            if flag_outputs:
                self.writer.add_histogram(f"{idx_to_name[i]}", preds[..., i], idx)
            if flag_postproc_outputs:
                self.writer.add_histogram(f"{idx_to_name[i]}_postproc", preds_denorm[..., i], idx)

    def on_after_backward(self, trainer, pl_module) -> None:
        log_grads = self._log_gradients and (trainer.global_step % self._log_gradients_freq == 0)
        self.log_parameters(pl_module.named_parameters(), trainer.global_step, log_grads=log_grads)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # if self.pl_module.global_rank != 0:
        #     return
        log_weights = self._log_weights and (self._log_weights_interval == "batch" and batch_idx % self._log_weights_freq == 0)
        log_clipped_grads = self._log_clipped_gradients and (batch_idx % self._log_clipped_gradients_freq == 0)
        self.log_parameters(
            pl_module.named_parameters(),
            trainer.global_step,
            log_weights=log_weights,
            log_clipped_grads=log_clipped_grads,
        )

        if self._log_preds and batch_idx % self._log_preds_freq == 0:
            self.log_preds(pl_module, outputs, trainer.global_step)

        flag_outputs = self._log_preds and batch_idx % self._log_preds_freq == 0
        flag_postproc_outputs = self._log_postproc_preds and batch_idx % self._log_postproc_preds_freq == 0

        self.log_preds(pl_module, outputs, trainer.global_step, flag_outputs, flag_postproc_outputs)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # if self.pl_module.global_rank != 0:
        #     return
        log_weights = self._log_weights and (
            self._log_weights_interval == "epoch" and (trainer.current_epoch + 1) % self._log_weights_freq == 0
        )
        self.log_parameters(pl_module.named_parameters(), trainer.current_epoch, log_weights=log_weights)


class ParentUUIDCallback(Callback):
    """A callback that retrieves the parent UUID for a model, if it is a child model."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the ParentUUIDCallback callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__()
        self.config = config

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: torch.nn.Module,
    ) -> None:
        del trainer  # unused
        pl_module.hparams["metadata"]["parent_uuid"] = checkpoint["hyper_parameters"]["metadata"]["uuid"]


class BaseLossMapPlot(BasePlotCallback):
    # Plot the accumulated loss over a given Map for the validation epoch at regular intervals
    def __init__(self, config, val_dset_len, **kwargs) -> None:
        super().__init__(config, val_dset_len, **kwargs)
        self.inner_subplots_kwargs = None
        self.counter = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if self.op_on_this_batch(batch_idx):
            self.accumulate(trainer, pl_module, outputs)
            self.counter += 1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._plot(trainer, pl_module)
        self.reset()
        self.counter = 0
        torch.cuda.empty_cache()

    def accumulate(self, trainer, pl_module, outputs) -> None:
        raise NotImplementedError

    def _plot(self, trainer, pl_module) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class AnemoiCheckpoint(ModelCheckpoint):
    """A checkpoint callback that saves the model after every validation epoch."""

    def __init__(self, config: OmegaConf, **kwargs: dict) -> None:
        """Initialise the AnemoiCheckpoint callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        kwargs : dict
            Additional keyword arguments for Pytorch ModelCheckpoint

        """
        super().__init__(**kwargs)
        self.config = config
        self.start = time.time()
        self._model_metadata = None
        self._tracker_metadata = None
        self._tracker_name = None

    @staticmethod
    def _torch_drop_down(trainer: pl.Trainer) -> torch.nn.Module:
        # Get the model from the DataParallel wrapper, for single and multi-gpu cases
        assert hasattr(trainer, "model"), "Trainer has no attribute 'model'! Is the Pytorch Lightning version correct?"
        return trainer.model.module.model if hasattr(trainer.model, "module") else trainer.model.model

    @rank_zero_only
    def model_metadata(self, model: torch.nn.Module) -> dict:
        if self._model_metadata is not None:
            return self._model_metadata

        self._model_metadata = {
            "model": model.__class__.__name__,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "summary": repr(
                torchinfo.summary(
                    model,
                    depth=50,
                    verbose=0,
                    row_settings=["var_names"],
                ),
            ),
        }

        return self._model_metadata

    def tracker_metadata(self, trainer: pl.Trainer) -> dict:
        if self._tracker_metadata is not None:
            return {self._tracker_name: self._tracker_metadata}

        if self.config.diagnostics.log.wandb.enabled:
            self._tracker_name = "wand"
            import wandb

            run = wandb.run
            if run is not None:
                self._tracker_metadata = {
                    "id": run.id,
                    "name": run.name,
                    "url": run.url,
                    "project": run.project,
                }
            return {self._tracker_name: self._tracker_metadata}

        if self.config.diagnostics.log.mlflow.enabled:
            self._tracker_name = "mlflow"

            from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

            mlflow_logger = next(logger for logger in trainer.loggers if isinstance(logger, AnemoiMLflowLogger))
            run_id = mlflow_logger.run_id
            run = mlflow_logger._mlflow_client.get_run(run_id)

            if run is not None:
                self._tracker_metadata = {
                    "id": run.info.run_id,
                    "name": run.info.run_name,
                    "url": run.info.artifact_uri,
                    "project": run.info.experiment_id,
                }
            return {self._tracker_name: self._tracker_metadata}

        return {}

    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        super()._remove_checkpoint(trainer, filepath)
        trainer.strategy.remove_checkpoint(self._get_inference_checkpoint_filepath(filepath))

    def _get_inference_checkpoint_filepath(self, filepath: str) -> str:
        """Defines the filepath for the inference checkpoint."""
        return Path(filepath).parent / Path("inference-" + str(Path(filepath).name))

    def _save_checkpoint(self, trainer: pl.Trainer, lightning_checkpoint_filepath: str) -> None:
        if trainer.is_global_zero:
            model = self._torch_drop_down(trainer)

            # We want a different uuid each time we save the model
            # so we can tell them apart in the catalogue (i.e. different epochs)
            checkpoint_uuid = str(uuid.uuid4())
            trainer.lightning_module._hparams["metadata"]["uuid"] = checkpoint_uuid

            trainer.lightning_module._hparams["metadata"]["model"] = self.model_metadata(model)
            trainer.lightning_module._hparams["metadata"]["tracker"] = self.tracker_metadata(trainer)

            trainer.lightning_module._hparams["metadata"]["training"] = {
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "elapsed_time": time.time() - self.start,
            }

            Path(lightning_checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)

            save_config = model.config
            model.config = None

            tmp_metadata = model.metadata
            model.metadata = None

            metadata = dict(**tmp_metadata)

            inference_checkpoint_filepath = self._get_inference_checkpoint_filepath(lightning_checkpoint_filepath)

            torch.save(model, inference_checkpoint_filepath)

            save_metadata(inference_checkpoint_filepath, metadata)

            model.config = save_config
            model.metadata = tmp_metadata

            self._last_global_step_saved = trainer.global_step

        trainer.strategy.barrier()

        # saving checkpoint used for pytorch-lightning based training
        trainer.save_checkpoint(lightning_checkpoint_filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = lightning_checkpoint_filepath

        if trainer.is_global_zero:
            from weakref import proxy

            # save metadata for the training checkpoint in the same format as inference
            save_metadata(lightning_checkpoint_filepath, metadata)

            # notify loggers
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class MemCleanUpCallback(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # This will be called after the ValidationCallback
        self.cleanup()

    def cleanup(self) -> None:
        torch.cuda.empty_cache()


class VideoPlotCallback(BasePlotCallback):
    """Evaluates the model performance over a (longer) rollout window."""

    # TODO: (Rilwan Adewoyin): Change this class so it plots the following videos:
    # Group 1 Video: Temporal evolution of the true target, predicted target and forecast error
    # 1. The temporal evolution of the true target (top left)
    # 2. The temporal evolution of the predicted target (top right)
    # 3. The temporal evolution of the forecast error (bottom left)
    # 4. A chart wihch shows x-axis time, y-axis aggregated & weighted error (true - predicted) across all latlons (e.g. effect of rollout length on error) (bottom right)

    # Group 2 Video: Temporal evolution of the true and predicted spectra
    # 5. The temporal evolution of the true and predicted spectra (spatial)

    def __init__(self, config, eval_dset_len) -> None:
        super().__init__(config, eval_dset_len)
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        self.eval_enabled = config.diagnostics.test.rollouteval.eval.enabled
        self.video_enabled = config.diagnostics.test.rollouteval.video.enabled
        self.eval_frequency = self._calculate_frequency(config.diagnostics.test.rollouteval.eval.frequency, eval_dset_len)
        self.video_frequency = self._calculate_frequency(config.diagnostics.test.rollouteval.video.frequency, eval_dset_len)
        self.video_rollout = config.diagnostics.test.rollouteval.video.rollout
        self.max_rollout = max(self.video_rollout, max(config.diagnostics.test.rollouteval.eval.rsteps_to_log, default=0))
        assert self.config.dataloader.batch_size.test == 1, "Batch size for testing must be 1!"

    def _calculate_frequency(self, frequency, eval_dset_len):
        if isinstance(frequency, float):
            return max(1, int(frequency * eval_dset_len))
        return frequency

    def _prepare_data(self, batch, pl_module):
        x = batch[:, 0 : pl_module.multi_step, ..., pl_module.data_indices.data.input.full]
        input_tensor_0 = batch[:, pl_module.multi_step - 1, ..., pl_module.data_indices.data.output.full].cpu()
        data_0 = self.post_processors(input_tensor_0).numpy()
        if pl_module.output_mask is not None:
            data_0[..., ~pl_module.output_mask, :] = np.nan
        return x, data_0

    def _plot_frame(self, ts, cache, dates, var_idx, var_name, lats, lons, vmin, vmax, path_dir):
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=400, subplot_kw={"projection": ccrs.PlateCarree()})
        fig.suptitle(f"{var_name} Rollout: {dates[ts].strftime('%Y-%m-%d %H:%M')}", fontsize=12)

        self._map_scatter(lats, lons, cache["y_proc"][ts, :, var_idx], cmap="viridis", axes=axs[0, 0], s=3, vmin=vmin, vmax=vmax)
        axs[0, 0].set_title("Target")

        self._map_scatter(lats, lons, cache["y_hat_proc"][ts, :, var_idx], cmap="viridis", axes=axs[0, 1], s=2, vmin=vmin, vmax=vmax)
        axs[0, 1].set_title("Prediction")

        error = cache["y_proc"][ts, :, var_idx] - cache["y_hat_proc"][ts, :, var_idx]
        self._map_scatter(lats, lons, error, cmap="viridis", axes=axs[1, 0], s=2, vmin=-vmax, vmax=vmax)
        axs[1, 0].set_title("Error")

        fig.tight_layout()
        frame_filename = f"{path_dir}/frame_{ts:04d}.png"
        plt.savefig(frame_filename)
        plt.close(fig)
        return frame_filename

    def _create_video(self, cache, dates, var_idx, var_name, lats, lons, path_dir):
        vmin = min(np.nanmin(cache["y_proc"][:, :, var_idx]), np.nanmin(cache["y_hat_proc"][:, :, var_idx]))
        vmax = max(np.nanmax(cache["y_proc"][:, :, var_idx]), np.nanmax(cache["y_hat_proc"][:, :, var_idx]))

        frame_filenames = [self._plot_frame(ts, cache, dates, var_idx, var_name, lats, lons, vmin, vmax, path_dir) for ts in range(cache["y_proc"].shape[0])]

        video_filename = Path(path_dir) / f"rollout_{var_name}.mp4"
        frame = cv2.imread(str(frame_filenames[0]))
        height, width, _layers = frame.shape
        video = cv2.VideoWriter(str(video_filename), cv2.VideoWriter_fourcc(*"mp4v"), 4, (width, height))

        for frame_filename in frame_filenames:
            video.write(cv2.imread(str(frame_filename)))
            Path(frame_filename).unlink()

        video.release()
        return str(video_filename)

    def _map_scatter(self, lats, lons, data, proj=None, cmap="viridis", axes=None, vmin=None, vmax=None, **kwargs):

        # for example for gaussian grid
        import cartopy.crs as ccrs

        if proj is None:
            proj = ccrs.PlateCarree()
        if axes is None:
            _, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(16, 6))
        else:
            ax = axes
        ax.coastlines()
        ax.gridlines()
        sc = ax.scatter(x=lons, y=lats, c=data, transform=proj, **kwargs, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sc)
        gl = ax.gridlines(draw_labels=True, alpha=0.3, color="grey", linewidth=0.5, linestyle="-", x_inline=False, y_inline=False)
        # gl.n_steps = 100
        gl.n_steps = 85
        gl.top_labels = False
        gl.right_labels = False
        return ax
