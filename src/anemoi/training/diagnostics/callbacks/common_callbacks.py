from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint, MemCleanUpCallback, WeightGradOutputLoggerCallback, ParentUUIDCallback, GraphTrainableFeaturesPlot
from anemoi.training.losses.utils import get_monitored_metric_name
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

def setup_checkpoint_callbacks(config, monitored_metrics):
    ckpt_callbacks = []
    if config.diagnostics.profiler:
        LOGGER.warning("Profiling is enabled - no checkpoints will be written!")
        return []

    for ckpt_cfg in config.diagnostics.checkpoints:
        filename, mode, dirpath = None, "max", None

        if ckpt_cfg.type == 'interval':
            dirpath = Path(config.hardware.paths.checkpoints) / next(k for k in ("every_n_train_steps", "train_time_interval", "every_n_epochs") if ckpt_cfg.get(k))
        elif ckpt_cfg.type == "performance":
            OmegaConf.set_readonly(ckpt_cfg, False)
            ckpt_cfg.kwargs['monitor'] = get_monitored_metric_name(monitored_metrics, ckpt_cfg['monitor'])
            OmegaConf.set_readonly(ckpt_cfg, True)
            monitor_name = f"perf_{ckpt_cfg.kwargs['monitor'].replace('/', '_')}"
            dirpath = Path(config.hardware.paths.checkpoints) / f"perf_{monitor_name}"
            # filename = f"epoch={{epoch}}-step={{step}}-{monitor_name}-{{{ckpt_cfg.kwargs['monitor']}}:.5f}"
            filename = "epoch={epoch:03d}-step={step:05d}-" + monitor_name + "-{" + ckpt_cfg.kwargs['monitor'] + ":.5f}"

        if hasattr(config.training, 'rollout'):
            from anemoi.training.diagnostics.callbacks.forecast import AnemoiCheckpointRollout
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
                    verbose=False
                )
            )
        else:
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
                    verbose=False,
                )
            )
    return ckpt_callbacks

def setup_early_stopping_callbacks(config, monitored_metrics):
    es_callbacks = []
    for es_config in config.diagnostics.early_stoppings:
        OmegaConf.set_readonly(es_config, False)
        es_config['monitor'] = get_monitored_metric_name(monitored_metrics, es_config.monitor)
        OmegaConf.set_readonly(es_config, True)

        if hasattr(config.training, 'rollout'):
            from anemoi.training.diagnostics.callbacks.forecast import EarlyStoppingRollout
            es_callbacks.append(
                EarlyStoppingRollout(
                    **es_config,
                    check_finite=True,
                    verbose=True,
                    strict=True,
                    log_rank_zero_only=True
                )
            )
        else:
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

def setup_convergence_monitoring_callbacks(config):
    cm_callbacks = []
    if any([config.diagnostics.log.wandb.enabled, config.diagnostics.log.mlflow.enabled]):
        cm_callbacks.append(LearningRateMonitor(logging_interval="step"))
    return cm_callbacks

def setup_model_averaging_callbacks(config):
    ma_callbacks = []
    if config.training.swa.enabled:
        ma_callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.training.swa.lr,
                swa_epoch_start=min(int(0.75 * config.training.max_epochs), config.training.max_epochs - 1),
                annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
                annealing_strategy="cos",
            )
        )
    return ma_callbacks

def get_common_callbacks(config: DictConfig, monitored_metrics) -> list:
    trainer_callbacks = []
    
    trainer_callbacks.extend(setup_checkpoint_callbacks(config, monitored_metrics))
    trainer_callbacks.extend(setup_early_stopping_callbacks(config, monitored_metrics))
    trainer_callbacks.extend(setup_convergence_monitoring_callbacks(config))
    trainer_callbacks.extend(setup_model_averaging_callbacks(config))
    
    trainer_callbacks.append(MemCleanUpCallback())
    trainer_callbacks.append(WeightGradOutputLoggerCallback())
    trainer_callbacks.append(ParentUUIDCallback(config))
    
    if config.diagnostics.plot.learned_features:
        LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
        trainer_callbacks.append(GraphTrainableFeaturesPlot(config))
    
    return trainer_callbacks
