# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
import math
import os
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.utils.data_structures import NestedTrainingSample
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch import nn

from anemoi.training.losses.mse import WeightedMSELoss
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.utils.jsonify import map_config_to_primitives

LOGGER = logging.getLogger(__name__)


class DummyEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.encoders["era5"] = nn.Linear(101, 64)
        self.encoders["metar"] = nn.Linear(14, 64)
        self.encoders["noaa-atms"] = nn.Linear(32, 64)

        self.mixer = nn.Linear(64, 64)

    def forward(self, x):
        y = {}

        assert set(x.keys()) == set(self.encoders.keys()), f"Keys do not match: {set(x.keys())} != {set(self.encoders.keys())}"

        for key in self.encoders.keys():
            encoder, xt = self.encoders[key], x[key]
            assert isinstance(xt, torch.Tensor), f"Expected tensor, got {type(xt)}"
            yt = encoder(xt)

            assert yt.shape[0] == 1, yt.shape
            yt = yt[0, :,:]

            y[key] = yt

        # return y
        y_as_list = [y[key] for key in self.encoders.keys()]
        from anemoi.utils.data_structures import str_
        print(str_(y_as_list))
        cat = torch.cat(y_as_list)
        return self.mixer(cat)


def get_class(class_name: str):
    import importlib

    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return module.__dict__[class_name]


cls = get_class("anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec")
print(cls)


def instantiate_(_target_: str, **kwargs):
    return get_class(_target_)(**kwargs)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information

        """
        super().__init__()

        graph_data = graph_data.to(self.device)

        self.model = AnemoiModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )
        self.dummy_model = DummyEncoderModel()
        self.dummy_model.train()
        

        self.data_indices = data_indices

        self.save_hyperparameters()

        # self.latlons_data = graph_data[config.graph.data].x # this line should move somewhere else anyway
        self.loss_weights = graph_data[config.graph.data][config.model.node_loss_weight].squeeze()

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        self.metric_ranges, self.metric_ranges_validation, loss_scaling = self.metrics_loss_scaling(
            config,
            data_indices,
        )
        self.loss = WeightedMSELoss(node_weights=self.loss_weights, data_variances=loss_scaling)
        self.metrics = WeightedMSELoss(node_weights=self.loss_weights, ignore_nans=True)

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.use_zero_optimizer = config.training.zero_optimizer

        self.model_comm_group = None

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.enable_plot = config.diagnostics.plot.enabled

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model
        self.model_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_model,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x, self.model_comm_group)

    @staticmethod
    def metrics_loss_scaling(config: DictConfig, data_indices: IndexCollection) -> tuple[dict, Tensor]:
        metric_ranges = defaultdict(list)
        metric_ranges_validation = defaultdict(list)
        loss_scaling = (
            np.ones((len(data_indices.internal_data.output.full),), dtype=np.float32)
            * config.training.loss_scaling.default
        )

        pressure_level = instantiate(config.training.pressure_level_scaler)

        LOGGER.info(
            "Pressure level scaling: use scaler %s with slope %.4f and minimum %.2f",
            type(pressure_level).__name__,
            pressure_level.slope,
            pressure_level.minimum,
        )

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges[f"pl_{split[0]}"].append(idx)
                # Create pressure levels in loss scaling vector
                if split[0] in config.training.loss_scaling.pl:
                    loss_scaling[idx] = config.training.loss_scaling.pl[split[0]] * pressure_level.scaler(
                        int(split[-1]),
                    )
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                metric_ranges[f"sfc_{key}"].append(idx)
                # Create surface variables in loss scaling vector
                if key in config.training.loss_scaling.sfc:
                    loss_scaling[idx] = config.training.loss_scaling.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges[key] = [idx]
        loss_scaling = torch.from_numpy(loss_scaling)
        # metric for validation, after postprocessing
        for key, idx in data_indices.model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1 and split[1].isdigit():
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges_validation[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges_validation[f"sfc_{key}"].append(idx)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges_validation[key] = [idx]
        return metric_ranges, metric_ranges_validation, loss_scaling

    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    def advance_field_input(
        self,
        x: Tensor,
        y_pred: Tensor,
        batch: Tensor,
        rollout_step: int,
    ) -> Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.internal_model.input.prognostic] = y_pred[
            ...,
            self.data_indices.internal_model.output.prognostic,
        ]

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.internal_model.input.forcing] = batch[
            -1
        ][  # TODO: do not hardcode ERA5 list index
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.internal_data.input.forcing,
        ]
        return x

    @cached_property
    def _encoders_keys(self):
        # TODO: make this flexible, read from config
        # or remove it and use the data_indices directly and no dictionary
        lst = ["era5", "metar", "noaa_20_atms"]  # this must match the order given in the datasets.

        def check_prefixes():
            for name, (i, j) in self.data_indices.name_to_index.items():
                for k in lst:
                    if name.startswith(k) and k != lst[i]:
                        raise ValueError(
                            f"Config error when indexing '{name}'. Index is {i, j}. The name starts with {k} but should start with {lst[i]} (list of keys is {lst}).",
                        )

        check_prefixes()

        print(f"✅ encoders keys are hard coded {lst}")
        return lst

    def _step(
        self,
        training_sample: Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        del batch_idx

        # TODO: Should we have a batch as input or a single sample?

        sample = NestedTrainingSample(training_sample, state_type="torch")
        del training_sample
        #sample.to('gpu')
        sample.to('cpu')

        print("Entering _step")
        print(f"🆗 Training sample = {sample}:")
        print()

        loss = torch.zeros(1, dtype=sample.dtype, device=self.device, requires_grad=False)
        # for validation not normalized in-place because remappers cannot be applied in-place
        sample = self.model.pre_processors(sample, in_place=not validation_mode)
        metrics = {}
        print(f"🆗 Normalisation done . exit here for now")

        # start rollout of preprocessed batch
        x = sample[0]
        y_ref = sample[1]
        y = self.dummy_model(x)

        y_preds = [y]
        print(f"🆗 _step done . exit here for now")
        exit()
        return loss, metrics, y_preds

        # x = batch[ :, 0 : self.multi_step, ..., self.data_indices.internal_data.input.full]  # (bs, multi_step, latlon, nvar)

        # y_preds = []
        # for rollout_step in range(self.rollout):
        #     # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
        #     y_pred = self(x)

        #     y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.internal_data.output.full]
        #     # y includes the auxiliary variables, so we must leave those out when computing the loss
        #     loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)

        #     x = self.advance_field_input(x, y_pred, batch, rollout_step)

        #     if validation_mode:
        #         metrics_next, y_preds_next = self.calculate_val_metrics(
        #             y_pred,
        #             y,
        #             rollout_step,
        #             enable_plot=self.enable_plot,
        #         )
        #         metrics.update(metrics_next)
        #         y_preds.extend(y_preds_next)

        # # scale loss
        # loss *= 1.0 / self.rollout
        # return loss, metrics, y_preds

    def calculate_val_metrics(
        self,
        y_pred: Tensor,
        y: Tensor,
        rollout_step: int,
        enable_plot: bool = False,
    ) -> tuple[dict, list]:
        metrics = {}
        y_preds = []
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)
        for mkey, indices in self.metric_ranges_validation.items():
            metrics[f"{mkey}_{rollout_step + 1}"] = self.metrics(
                y_pred_postprocessed[..., indices],
                y_postprocessed[..., indices],
            )

        if enable_plot:
            y_preds.append(y_pred)
        return metrics, y_preds

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            "train_wmse",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)
        self.log(
            "val_wmse",
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
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss, y_preds

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        if self.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=self.lr,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                betas=(0.9, 0.95),
                lr=self.lr,
            )  # , fused=True)

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
