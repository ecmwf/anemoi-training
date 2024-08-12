import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(config: DictConfig):
    if not config.diagnostics.log.mlflow.enabled:
        LOGGER.debug("MLFlow logging is disabled.")
        return None

    from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
    from anemoi.training.diagnostics.mlflow.logger import get_mlflow_run_params

    resumed = config.training.run_id is not None
    forked = config.training.fork_run_id is not None

    save_dir = config.hardware.paths.logs.mlflow
    tracking_uri = config.diagnostics.log.mlflow.tracking_uri
    offline = config.diagnostics.log.mlflow.offline

    if (resumed or forked) and (offline):  # when resuming or forking offline -
        # tracking_uri = ${hardware.paths.logs.mlflow}
        tracking_uri = save_dir
    # create directory if it does not exist
    Path(config.hardware.paths.logs.mlflow).mkdir(parents=True, exist_ok=True)
    run_id, run_name, tags = get_mlflow_run_params(config, tracking_uri)

    log_hyperparams = True
    if resumed and not config.diagnostics.log.mlflow.on_resume_create_child:
        LOGGER.info(
            "Resuming run without creating child run - MLFlow logs will not update the\
                initial runs hyperparameters with those of the resumed run.\
                To update the initial run's hyperparameters, set `diagnostics.log.mlflow.on_resume_create_child: True`.",
        )
        log_hyperparams = False

    logger = AnemoiMLflowLogger(
        experiment_name=config.diagnostics.log.mlflow.experiment_name,
        tracking_uri=tracking_uri,
        save_dir=save_dir,
        run_name=run_name,
        run_id=run_id,
        log_model=config.diagnostics.log.mlflow.log_model,
        offline=offline,
        tags=tags,
        resumed=resumed,
        forked=forked,
        log_hyperparams=log_hyperparams,
        authentication=config.diagnostics.log.mlflow.authentication,
    )
    config_params = OmegaConf.to_container(config, resolve=True)

    logger.log_hyperparams(config_params)

    if config.diagnostics.log.mlflow.terminal:
        logger.log_terminal_output(artifact_save_dir=config.hardware.paths.plots)
    if config.diagnostics.log.mlflow.system:
        logger.log_system_metrics()

    return logger


def get_tensorboard_logger(config: DictConfig) -> Optional[pl.loggers.TensorBoardLogger]:
    """Setup TensorBoard experiment logger.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    Optional[pl.loggers.TensorBoardLogger]
        Logger object, or None
    """
    if not config.diagnostics.log.tensorboard.enabled:
        LOGGER.debug("Tensorboard logging is disabled.")
        return None

    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger(
        save_dir=config.hardware.paths.logs.tensorboard,
        log_graph=False,
    )
    return logger


def get_wandb_logger(config: DictConfig, model: pl.LightningModule) -> Optional[pl.loggers.WandbLogger]:
    """Setup Weights & Biases experiment logger.

    Parameters
    ----------
    config : DictConfig
        Job configuration
    model: GraphForecaster
        Model to watch

    Returns
    -------
    Optional[pl.loggers.WandbLogger]
        Logger object
    """
    if not config.diagnostics.log.wandb.enabled:
        LOGGER.debug("Weights & Biases logging is disabled.")
        return None

    try:
        from pytorch_lightning.loggers.wandb import WandbLogger
    except ImportError as err:
        raise ImportError("To activate W&B logging, please install `wandb` as an optional dependency.") from err

    logger = WandbLogger(
        project="aifs-fc",
        entity="ecmwf-ml",
        id=config.training.run_id,
        save_dir=config.hardware.paths.logs.wandb,
        offline=config.diagnostics.log.wandb.offline,
        log_model=config.diagnostics.log.wandb.log_model,
        resume=config.training.run_id is not None,
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    if config.diagnostics.log.wandb.gradients or config.diagnostics.log.wandb.parameters:
        if config.diagnostics.log.wandb.gradients and config.diagnostics.log.wandb.parameters:
            log_ = "all"
        elif config.diagnostics.log.wandb.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=config.diagnostics.log.interval, log_graph=False)

    return logger
