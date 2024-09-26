# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from hydra.utils import instantiate

from anemoi.training.diagnostics.callbacks import plotting
from anemoi.training.diagnostics.callbacks.checkpointing import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks.evaluation import RolloutEval
from anemoi.training.diagnostics.callbacks.id import ParentUUIDCallback
from anemoi.training.diagnostics.callbacks.learning_rate import LearningRateMonitor

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)


# Callbacks to add according to flags in the config
CONFIG_ENABLED_CALLBACKS: list[tuple[list[str] | str, list[type[Callback]] | type[Callback]]] = [
    (["diagnostics.log.wandb.enabled", "diagnostics.log.mlflow.enabled"], LearningRateMonitor),
    ("diagnostics.eval.enabled", RolloutEval),
    (
        "diagnostics.plot.enabled",
        [
            plotting.PlotLoss,
            plotting.PlotSample,
        ],
    ),
    ("diagnostics.plot.learned_features", plotting.GraphTrainableFeaturesPlot),
]


def _get_checkpoint_callback(config: DictConfig) -> list[AnemoiCheckpoint] | None:
    """Get checkpointing callback"""
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

    if not config.diagnostics.profiler:
        for save_key, (name, save_frequency, save_n_models) in ckpt_frequency_save_dict.items():
            if save_frequency is not None:
                LOGGER.debug("Checkpoint callback at %s = %s ...", save_key, save_frequency)
                return (
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
                    ]
                )
            else:
                LOGGER.debug("Not setting up a checkpoint callback with %s", save_key)
    else:
        # the tensorboard logger + pytorch profiler cause pickling errors when writing checkpoints
        LOGGER.warning("Profiling is enabled - will not write any training or inference model checkpoints!")
    return None


def _get_config_enabled_callbacks(config: DictConfig) -> list[Callback]:
    """Get callbacks that are enabled in the config as according to CONFIG_ENABLED_CALLBACKS

    Provides backwards compatibility
    """
    callbacks = []

    def nestedget(conf: DictConfig, key, default):
        keys = key.split(".")
        for k in keys:
            conf = conf.get(k, default)
            if not isinstance(conf, dict):
                break
        return conf

    for enable_key, callback_list in CONFIG_ENABLED_CALLBACKS:
        if isinstance(enable_key, list):
            if not any(nestedget(config, key, False) for key in enable_key):
                continue
        elif not nestedget(config, enable_key, False):
            continue

        if isinstance(callback_list, list):
            callbacks.extend(map(lambda x: x(config), callback_list))
        else:
            callbacks.append(callback_list(config))

    if config.diagnostics.plot.enabled:
        if (config.diagnostics.plot.parameters_histogram or config.diagnostics.plot.parameters_spectrum) is not None:
            callbacks.append(plotting.PlotAdditionalMetrics(config))
        if config.diagnostics.plot.get("longrollout") and config.diagnostics.plot.longrollout.enabled:
            callbacks.append(plotting.LongRolloutPlots(config))

    return callbacks


def get_callbacks(config: DictConfig) -> list[Callback]:  # noqa: C901
    """Setup callbacks for PyTorch Lightning trainer.

    Set `config.diagnostics.callbacks` to a list of callback configurations
    in hydra form.

    E.g.:
    ```
    callbacks:
        swa: _target_: pytorch_lightning.callbacks.stochastic_weight_avg.StochasticWeightAveraging
                swa_lr: 1e-4
                swa_epoch_start: 123
                annealing_epochs: 5
                annealing_strategy: cos
                device: null
    ```

    Set `config.diagnostics.plot_callbacks` to a list of plotting callback configurations
    will only be added if `config.diagnostics.plot.enabled` is set to True.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    List[Callback]
        A list of PyTorch Lightning callbacks

    """

    trainer_callbacks: list[Callback] = []

    # Get Checkpoint callback
    checkpoint_callback = _get_checkpoint_callback(config)
    if checkpoint_callback is not None:
        trainer_callbacks.extend(checkpoint_callback)

    # Base callbacks
    for callback in config.diagnostics.get("callbacks", []):
        # Instantiate new callbacks
        trainer_callbacks.append(instantiate(callback))

    # Plotting callbacks
    if config.diagnostics.plot.enabled:
        for callback in config.diagnostics.get("plot_callbacks", []):
            # Instantiate new callbacks
            trainer_callbacks.append(instantiate(callback))

    # Extend with backward compatible callbacks
    trainer_callbacks.extend(_get_config_enabled_callbacks(config))

    # Parent UUID callback
    trainer_callbacks.append(ParentUUIDCallback(config))

    return trainer_callbacks


__all__ = ["get_callbacks", "RolloutEval", "LearningRateMonitor", "plotting"]
