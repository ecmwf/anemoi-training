import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
from collections import defaultdict

from timm.scheduler import CosineLRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer
import logging
from torch.nn import ModuleList
from hydra.utils import instantiate
from anemoi.utils.config import DotDict
from anemoi.training.utils.jsonify import map_config_to_primitives
from torch_geometric.data import HeteroData


from omegaconf import OmegaConf
from typing import Optional

LOGGER = logging.getLogger(__name__)
from anemoi.models.data_indices.collection import IndexCollection


class AnemoiLightningModule(pl.LightningModule):
    """Base class for Anemoi Lightning Modules (Forecasting and Reconstruction)."""

    def __init__(self, config: DictConfig, graph_data: HeteroData, statistics: dict, statistics_tendencies: Optional[dict], data_indices: dict, metadata: dict, model_cls):
        super().__init__()

        graph_data = graph_data.to(self.device)

        # Initialize the model (either Forecasting or Reconstruction)
        self.model = model_cls(
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        self.data_indices = data_indices
        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x
        self.node_weights = graph_data[config.graph.data][config.model.node_loss_weight].squeeze()

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices)
        self.feature_weights = self.get_feature_weights(config, data_indices)
        self.multi_step = config.training.multistep_input

        # Loss and metrics initialization
        self.loss = instantiate(
            config.training.loss,
            config.training,
            node_weights=self.node_weights,
            latent_node_weights=self.get_latent_weights(graph_data, config),
            feature_weights=self.feature_weights,
            data_indices_model_output=self.data_indices.model.output,
        )

        self.val_metrics = ModuleList(
            [
                instantiate(
                    vm_cfg,
                    node_weights=self.node_weights,
                    latent_weights=self.get_latent_weights(graph_data, config),
                    feature_weights=self.feature_weights,
                    data_indices_model_output=self.data_indices.model.output,
                )
                for vm_cfg in config.training.val_metrics
            ],
        )

        self.config = config
        self.setup_communication(config)

    @staticmethod
    def get_val_metric_ranges(config: DictConfig, data_indices: IndexCollection) -> dict:

        # NOTE: This a grouping by pressure level
        val_metric_ranges = defaultdict(list)

        for key, idx in data_indices.model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1:
                # Group metrics for pressure levels (e.g., Q, T, U, V, etc.)
                val_metric_ranges[f"pl_{split[0]}"].append(idx)

                if key in config.training.metrics or "all_individual" in config.training.metrics:
                    val_metric_ranges[key] = [idx]
            else:
                val_metric_ranges["sfc"].append(idx)

            # Specific features to calculate metrics for
            if key in config.training.metrics:
                val_metric_ranges[key] = [idx]

        if "all_grouped" in config.training.metrics:
            val_metric_ranges.update(
                {"all" : list(data_indices.model.output.name_to_index.values())},
            )

        return val_metric_ranges

    def get_feature_weights(self, config: DictConfig, data_indices: IndexCollection) -> torch.Tensor:
        """
        Calculates the feature weights for each output feature based on the configuration, data indices. User can specify weighting strategies based on pressure level, feature type, and inverse variance scaling. Any strategies provided are combined.

        Parameters
        ----------
        config (DictConfig): A configuration object that contains the training parameters.
        data_indices (IndexCollection): An object that contains the indices of the data.

        Returns
        -------
        torch.Tensor: A tensor that contains the calculates weights for the feature dimension during loss computation.

        """
        feature_weights = np.ones((len(data_indices.data.output.full),), dtype=np.float32) * config.training.feature_weighting.default
        pressure_level = instantiate(config.training.pressure_level_scaler)

        LOGGER.info(
            "Pressure level scaling: use scaler %s with slope %.4f and minimum %.2f",
            type(pressure_level).__name__,
            pressure_level.slope,
            pressure_level.minimum,
        )

        for key, idx in data_indices.model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1:
                # Apply pressure level scaling
                if split[0] in config.training.feature_weighting.pl:
                    feature_weights[idx] = config.training.feature_weighting.pl[split[0]] * pressure_level.scaler(int(split[1]))
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                # Apply surface variable scaling
                if key in config.training.feature_weighting.sfc:
                    feature_weights[idx] = config.training.feature_weighting.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)

        if config.training.feature_weighting.inverse_tendency_variance_scaling:
            variances = self.model.statistics_tendencies["stdev"][data_indices.data.output.full]
            feature_weights /= variances

        return torch.from_numpy(feature_weights)

    def configure_optimizers(self):

        if self.config.training.optimizer.scale_by_gpus:
            lr = (
                (self.config.hardware.num_nodes
                    * self.config.hardware.num_gpus_per_node
                    * self.config.training.optimizer.lr)
                    / self.config.hardware.num_gpus_per_model
                )
        else:
            lr = self.config.training.optimizer.lr

        if self.config.training.zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=lr,
                weight_decay=self.config.training.optimizer.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                betas=(0.9, 0.95),
                lr=lr,
                weight_decay=self.config.training.optimizer.weight_decay,
            )

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.config.training.scheduler.lr_min,
            t_initial=self.training_steps(),
            warmup_t=self.config.training.scheduler.warmup_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_steps(self):
        if self.config.training.max_steps is not None:
            return self.config.training.max_steps
        train_batches_per_epoch = len(self.trainer.datamodule.ds_train) // self.config.dataloader.batch_size["training"]
        return self.config.training.max_epochs * train_batches_per_epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)

    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        scheduler.step(epoch=self.trainer.global_step)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx, validation_mode=False)    
        self.log(
            f"train/loss/{self.loss.name}",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        return train_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # TODO: change this to use the name from the loss function to avoid hardcoding
        with torch.no_grad():
            val_loss, metrics, outputs = self._step(batch, batch_idx, validation_mode=True)
        
        self.log(
            f"val/loss/{self.loss.name}",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        for mname, mvalue in metrics.items():
            self.log(
                "val/" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss, outputs

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch = self.model.pre_processors_state(batch, in_place=False)
            x = batch[:, :, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, inp_ens, multi_step, latlon, nvar)
            y_hat = self(x)

        return self.model.post_processors_state(y_hat, in_place=False)