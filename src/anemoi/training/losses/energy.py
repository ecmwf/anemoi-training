# Copyright (c) 2023 Your Company Name

from __future__ import annotations

import os
import types
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor
from anemoi.training.losses.utils import process_file
import logging
from anemoi.training.losses.kcrps import MultivariatekCRPS, GroupedMultivariatekCRPS
if TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

LOGGER = logging.getLogger(__name__)

class EnergyScore(MultivariatekCRPS):
    """EnergyScore loss for ensemble forecasts.

    #TODO (rilwan-ade): allow the distance metric to be passed as an argument 
    
    Attributes
    ----------
    node_weights : torch.Tensor
        Weights for different areas in the loss calculation.
    feature_weights : Optional[torch.Tensor]
        Scaling factors applied to the loss for each feature.
    group_on_dim : int
        Dimension to group on for the loss calculation (-1 for feature, -2 for spatial, -3 for temporal).
    beta : torch.Tensor or int
        Power exponent for the energy loss calculation.
    fair : bool
        If True, apply a fair version of the energy score, else use the standard version.
    ignore_nans : Optional[bool]
        Flag to determine whether to ignore NaN values during calculations.
    logging : str
        Logging level for the class.

    A class to compute the Energy Score (ES) for evaluating multivariate probabilistic forecasts.

    The **Energy Score (ES)** is a proper scoring rule used to measure the distance between a
    forecast distribution and the observed distribution in a multivariate space.

    The formula for the Energy Score is:

    .. math::

        \text{ES}(F, G) = 2 \mathbb{E}[d(X, Y)] - \mathbb{E}[d(X, X')] - \mathbb{E}[d(Y, Y')]

    where:
        - \( X, X' \sim F \) are samples from the forecast distribution,
        - \( Y, Y' \sim G \) are samples from the observed distribution,
        - \( d(x, y) \) is a distance metric (e.g., Euclidean distance).

    The Energy Score focuses on evaluating the full multivariate distribution and is ideal for
    assessing models that produce probabilistic forecasts in multiple dimensions.
    """
    def __init__(self, node_weights: torch.Tensor, feature_weights: torch.Tensor | None, group_on_dim: int = -1, beta: torch.Tensor | float = 1.0, fair: bool = True, ignore_nans: bool | None = False, logging: str = "INFO", **kwargs) -> None:

        p_norm = 2.0

        assert beta > 0, "Power must be positive"
        assert beta <= 2, "Power must be less than 2"

        super().__init__(node_weights, feature_weights, fair, p_norm, beta, implementation="vectorized", ignore_nans=ignore_nans, group_on_dim=group_on_dim, **kwargs)

    @cached_property
    def name(self) -> str:
        """Generate a log name based on parameters."""
        power_str = f"p{format(self.beta.item(), '.2g')}" if torch.is_tensor(self.beta) else f"p{format(self.beta, '.2g')}"
        fair_str = "f" if self.fair else ""
        return f"{fair_str}energy_d{self.group_on_dim}_{power_str}"


class GroupedEnergyScore(GroupedMultivariatekCRPS):
    """Grouped Energy Score using Voronoi patches or grouping strategies

    This class computes the Energy score over spatial patches defined by Voronoi regions. It includes
    efficiency improvements through operation batching and the use of CUDA streams.

    Attributes
    ----------
        options(List[str]): List of available Voronoi patch options.
        node_weights(Tensor): Tensor of area weights for the Energy score computation.

        beta(Optional[Union[Tensor, float]]): Power parameter for the Energy score computation. Defaults to 1.0.False.

        op_batching(int): Number of operations to batch together for improved efficiency. Defaults to 1 to disable batching.
        num_patch_groups(int): Number of patch groups.
        patch_set_lens(Tensor): Tensor containing the lengths of each patch group.
        streams(Optional[List[torch.cuda.Stream]]): List of CUDA streams for parallel computation.

    Efficiency Improvements:
        1. Operation Batching:
            - When `op_batching` > 1, similar-sized patches are batched together for simultaneous processing.
            - This reduces the number of separate operations, leading to fewer kernel launches and better GPU utilization.

    Methods
    -------
        __init__(self, node_weights: Tensor, li_patch_set: List[dict[str, list[int]]], feature_weights: Optional[Tensor] = None,
                 beta: Optional[Union[Tensor, float]] = 1.0, random_patches: bool = False,
                 cuda_stream_count: int = 1, op_batching: int = 1, **kwargs) -> None:
            Initializes the GroupedEnergyScore_voronoi instance.

        forward(self, preds: Tensor, target: Tensor, squash: bool = True, enum_feature_weight_scalenorm: int = 0,
                enum_area_weight_scalenorm: int = 0, indices_adjusted=None) -> Tensor:
            Forward pass of the Grouped Energy Score computation.

        name(self) -> str:
            Cached property to generate a log name based on the beta and group dimension.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor,
        patch_method: str,
        beta: float | None = 1.0,
        group_on_dim: int = -1,
        fair: bool = True,
        logging: str = "INFO",
        data_indices_model_output: DictConfig | None = None,
        op_batching: int = 1,
        li_patch_set: list[dict[str, list[int]]] | None = None,
        patches_dir: Path | str | None = None,
        implementation: str = "vectorized",
        **kwargs
    ) -> None:
        p_norm: float = 2.0

        super().__init__(
            node_weights=node_weights,
            feature_weights=feature_weights,
            patch_method=patch_method,
            p_norm=p_norm,
            beta=beta,
            group_on_dim=group_on_dim,
            fair=fair,
            logging=logging,
            data_indices_model_output=data_indices_model_output,
            op_batching=op_batching,
            li_patch_set=li_patch_set,
            patches_dir=patches_dir,
            implementation=implementation
        )

    @cached_property
    def name(self) -> str:

        power_str = "b" + format(self.beta.item(), ".2g") if torch.is_tensor(self.beta) else "b" + format(self.beta, ".2g")

        patch_method_name = self._options_short_name.get(self.patch_method, self.patch_method)

        f_str = "f" if self.fair else ""

    
        return f"{f_str}penergy_{patch_method_name}_{power_str}"

