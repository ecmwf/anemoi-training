# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings

        """
        super().__init__()

        LOGGER.debug(
            "Setting up RolloutEval callback with rollout = %d, frequency = %d ...",
            config.diagnostics.eval.rollout,
            config.diagnostics.eval.frequency,
        )
        self.rollout = config.diagnostics.eval.rollout
        self.frequency = config.diagnostics.eval.frequency

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        metrics = {}

        # start rollout
        batch = pl_module.model.pre_processors(batch, in_place=False)
        x = batch[
            :,
            0 : pl_module.multi_step,
            ...,
            pl_module.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        assert (
            batch.shape[1] >= self.rollout + pl_module.multi_step
        ), "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            for rollout_step in range(self.rollout):
                y_pred = pl_module(x)  # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
                y = batch[
                    :,
                    pl_module.multi_step + rollout_step,
                    ...,
                    pl_module.data_indices.internal_data.output.full,
                ]  # target, shape = (bs, latlon, nvar)
                # y includes the auxiliary variables, so we must leave those out when computing the loss
                loss += pl_module.loss(y_pred, y)

                x = pl_module.advance_input(x, y_pred, batch, rollout_step)

                metrics_next, _ = pl_module.calculate_val_metrics(y_pred, y, rollout_step)
                metrics.update(metrics_next)

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch.shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:
        pl_module.log(
            f"val_r{self.rollout}_wmse",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            batch_size=bs,
            sync_dist=False,
            rank_zero_only=True,
        )
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"val_r{self.rollout}_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=pl_module.logger_enabled,
                batch_size=bs,
                sync_dist=False,
                rank_zero_only=True,
            )

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del outputs  # outputs are not used
        if batch_idx % self.frequency == 0:
            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = torch.autocast(device_type=batch.device.type, dtype=dtype) if dtype is not None else nullcontext()

            with context:
                self._eval(pl_module, batch)
