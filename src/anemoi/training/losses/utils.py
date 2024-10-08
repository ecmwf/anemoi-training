# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging

import torch
from torch import nn

from typing import TYPE_CHECKING
from anemoi.models.distributed.graph import gather_tensor
import einops
import numpy as np
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
LOGGER = logging.getLogger(__name__)


def grad_scaler(
    module: nn.Module,
    grad_in: tuple[torch.Tensor, ...],
    grad_out: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | None:
    """Scales the loss gradients.

    Uses the formula in https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2

    Use <module>.register_full_backward_hook(grad_scaler, prepend=False) to register this hook.

    Parameters
    ----------
    module : nn.Module
        Loss object (not used)
    grad_in : tuple[torch.Tensor, ...]
        Loss gradients
    grad_out : tuple[torch.Tensor, ...]
        Output gradients (not used)

    Returns
    -------
    tuple[torch.Tensor, ...]
        Re-scaled input gradients

    """
    del module, grad_out
    # first grad_input is that of the predicted state and the second is that of the "ground truth" (== zero)
    channels = grad_in[0].shape[-1]  # number of channels
    channel_weights = torch.reciprocal(torch.sum(torch.abs(grad_in[0]), dim=1, keepdim=True))  # channel-wise weights
    new_grad_in = (
        (channels * channel_weights) / torch.sum(channel_weights, dim=-1, keepdim=True) * grad_in[0]
    )  # rescaled gradient
    return new_grad_in, grad_in[1]


def gather_and_compute_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    gather_matrix: torch.Tensor,
    loss: torch.nn.Module,
    ens_comm_group_size: int,
    ens_comm_group: ProcessGroup,
    return_pred_ens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Gather the ensemble members from all devices in my group.

    Eliminate duplicates (if any) and compute the loss.

    Args:
        y_pred: torch.Tensor
            Predicted state tensor, calculated on self.device
        y: torch.Tensor
            Ground truth
        gather_matrix: torch.Tensor
            Matrix is used to average the contributions of individual ensemble members
            gathered in the ensemble comm group
        loss: torch.nn.Module
            Loss function
        ens_comm_group_size: int
            Size of ensemble communication group
        ens_comm_group: int
            Process ensemble group
        return_pred_ens: bool
            Validation flag: if True, we return the predicted ensemble (post-gather)

    Returns
    -------
        loss_inc:
            Loss
        y_pred_ens:
            Predictions if validation mode
    """
    # step 1/ gather among all GPUs in the same ensemble group
    y_pred_ens = gather_tensor(y_pred, dim=1, shapes=[y_pred.shape] * ens_comm_group_size, mgroup=ens_comm_group)

    # step 2/ prune ensemble to get rid of the duplicates (if any) - uses the pre-built ensemble averaging matrix
    assert gather_matrix is not None

    y_pred_ens = einops.rearrange(y_pred_ens, "bs e latlon v -> bs v latlon e")  # ensemble dim must come last
    y_pred_ens = y_pred_ens @ gather_matrix
    y_pred_ens = einops.rearrange(y_pred_ens, "bs v latlon e -> bs e latlon v")  # reshape back to what it was

    # step 3/ compute the loss
    loss_inc = checkpoint(loss, y_pred_ens, y, squash=True, use_reentrant=False)

    # during validation, we also return the pruned ensemble (from step 2) so we can run diagnostics
    # an explicit cast is needed when running in mixed precision (i.e. with y_pred_ens.dtype == torch.(b)float16)
    return loss_inc, y_pred_ens.to(dtype=y.dtype) if return_pred_ens else None


def process_file(file_path):
    npz_file = np.load(file_path, fix_imports=False)
    return [npz_file[k] for k in npz_file]


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def get_monitored_metric_name(monitored_metrics, target_metric_name):
    assert (
        target_metric_name == "default" or target_metric_name in monitored_metrics
    ), f"""Monitored value:={target_metric_name} must either be the loss function,
                or a stated validation metric!"""

    if target_metric_name == "default":
        target_metric_name = next((mname for mname in monitored_metrics if mname.startswith("val/loss")), None)

        assert (
            target_metric_name is not None
        ), f"Default monitor value not found in monitored metrics: {monitored_metrics}"

    return target_metric_name

