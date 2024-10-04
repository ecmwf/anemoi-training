import os
from typing import Optional

import torch
from torch import Tensor, nn
import types
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from anemoi.training.losses.utils import process_file
from typing import TYPE_CHECKING
import einops
from omegaconf.dictconfig import DictConfig
import logging
LOGGER = logging.getLogger(__name__)


class kCRPS(nn.Module): 
    """Area-weighted kernel CRPS loss.
    
    A class to compute the Kernel Continuous Ranked Probability Score (KCRPS) for evaluating 
    univariate or marginal distributions.

    The **Kernel CRPS (KCRPS)** is a proper scoring rule that compares probabilistic forecasts 
    with observed values, typically using kernel density methods.

    The formula for the Kernel CRPS is:

    .. math::

        \text{KCRPS}(F, G) = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} k(x_i, x_j) 
                             - \frac{2}{nm} \sum_{i=1}^{n} \sum_{j=1}^{m} k(x_i, y_j)

    where:
        - \( x_i \) and \( x_j \) are samples from the forecast distribution \(F\),
        - \( y_i \) and \( y_j \) are samples from the observed distribution \(G\),
        - \( k(x, y) \) is a kernel function that measures similarity between two points.

    This score works well for evaluating marginal distributions and can be extended to 
    multivariate contexts.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: Optional[torch.Tensor],
        fair: bool = True,
        implementation: str = "low_mem",
        p_norm: float = 1.0,
        ignore_nans: Optional[bool] = False,
        **kwargs
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Args:
        node_weights : torch.Tensor
            Weights by area
        feature_weights : Optional[torch.Tensor], optional
            Loss weighting by feature
        fair: calculate a "fair" (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__()

        self.fair = fair

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)

        self.register_buffer("p_norm", torch.as_tensor(p_norm), persistent=False)

        self.implementation = implementation

        self._kernel_crps_impl = {"low_mem": self._kernel_crps_low_mem, "vectorized": self._kernel_crps_vectorized}

        # p_norm = 1 for L1 norm, p_norm = 2 for L2 norm
        # p_norm = 1 is the same as the standard CRPS
        self.p_norm = p_norm

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._kernel_crps_impl[self.implementation](preds, targets)

    def _kernel_crps_vectorized(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

            # Note this implementation is generalized to the case where y is not a single point but an ensemble

        Args:
            preds: predicted ensemble, shape (batch_size, ens_input, timesteps, latlon, n_vars)
            targets: ground truth, shape (batch_size, ens_target, timesteps, latlon, n_vars)

        Returns
        -------
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """
        ens_input = preds.shape[1]
        ens_target = targets.shape[1]
        # error = torch.mean(torch.abs(targets.unsqueeze(1) - preds), dim=1)
        # ensure that kCRPS using self.p_norm when doing the norm difference but not in a multivariate way 
        # e.g. do not do L1 norm just mae or do not do L2 just mse
        # error = torch.mean( torch.abs(targets - preds)**self.p_norm, dim=1 )

        error = torch.mean( torch.abs( targets.unsqueeze(1) - preds.unsqueeze(2) )**self.p_norm, dim=(1, 2) )

        if ens_input == 1:
            return error

        # Pred ensemble variance term
        coef = -0.5 / (ens_input * (ens_input - 1)) if self.fair else -0.5 / (ens_input**2)
        pairwise_diffs = torch.abs(preds.unsqueeze(1) - preds.unsqueeze(2))
        ens_var = pairwise_diffs.sum(dim=(1, 2)) * coef

        if ens_target != 1:
            # Target ensemble variance term
            coef = -0.5 / (ens_target * (ens_target - 1)) if self.fair else -0.5 / (ens_target**2)
            pairwise_diffs = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(2))
            ens_var += pairwise_diffs.sum(dim=(1, 2)) * coef

        return error + ens_var

    def _kernel_crps_low_mem(self, preds: torch.Tensor, targets: torch.Tensor, fair: bool = True) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Args:
            preds: predicted ensemble, shape (batch_size, ens_size, timesteps, latlon, n_vars)
            targets: ground truth, shape (batch_size, timesteps, latlon, n_vars)
            fair: unbiased ensemble variance calculation
        Returns:
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """
        preds = einops.rearrange(preds, "b e t l n -> b t l n e")

        ens_input = preds.shape[-1]
        ens_target = targets.shape[-1]

        error = torch.mean( torch.abs(targets - preds)**self.p_norm, dim=1 )

        if ens_input == 1:
            return error

        coef = -1.0 / (ens_input * (ens_input - 1)) if self.fair else -1.0 / (ens_input**2)
        
        ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
        for i in range(ens_input):  # loop version to reduce memory usage
            ens_var += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)
        ens_var = coef * ens_var

        if ens_target != 1:
            # Target ensemble variance term
            for i in range(ens_target):
                ens_var += (torch.sum(torch.abs(targets[..., i].unsqueeze(-1) - targets[..., i + 1 :]), dim=-1) * coef)

        return error + ens_var

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_scaling: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted kernel CRPS loss.

        Args:
            pred: predicted ensemble, shape (batch_size, nens_input, timesteps, latlon, n_vars)
            target: ground truth, shape (batch_size, nens_target, timesteps, latlon, n_vars)
            squash: bool, optional
                Reduce the spatial and feature dimensions
            feature_scaling: bool, optional
                Scale the loss by the feature weights
        feature_indices: indices of the features to scale the loss by

        Returns
        -------
            Weighted kernel CRPS loss
        """
        # Apply feature scaling
        if feature_scaling:
            if feature_indices is None:
                pred = pred * self.feature_weights / self.feature_weights.numel()
                target = target * self.feature_weights / self.feature_weights.numel()
            else:
                pred = pred * self.feature_weights[..., feature_indices] / self.feature_weights.numel()
                target = target * self.feature_weights[..., feature_indices] / self.feature_weights.numel()

        # Calculate kernel CRPS
        kcrps = self._kernel_crps(pred, target)

        # Apply node (spatial) weights
        kcrps *= (self.node_weights / self.sum_function(self.node_weights))

        # Squash (reduce spatial and feature dimensions)
        if squash:
            kcrps = self.sum_function(kcrps, dim=(-3, -2, -1)) 

        return kcrps.mean(dim=0)  # (timestep) or (timestep, latlon, nvar)

    def name(self) -> str:
        """Generate a log name based on parameters."""
        fair_str = "fair" if self.fair else ""

        kernel_str  = f"pnorm_{self.p_norm}"

        return f"{fair_str}kcrps_{kernel_str}"

class MultivariatekCRPS(nn.Module):
    """Multivariate kernel CRPS using vector norms across variables.

    NOTE: This is just kernel CRPS but here the kernel is L1 norm / the multivariate version of error CRPS

    This should be able to just subclass the LpCRPS class above

    TODO(rilwan-ade): Think of way to merge this with the LpCRPS class above using a multivatiate flag. if false then we just average across variable members at the end, if true we compute the norm earlier
    
    """

    def __init__(self, node_weights: torch.Tensor, feature_weights: Optional[torch.Tensor] = None, fair: bool = True, p_norm: int = 1.5, beta: float = 1.0, implementation: str = "vectorized", ignore_nans: Optional[bool] = False, group_on_dim: int = -1, **kwargs) -> None:
        """
        Args:
            node_weights: Tensor of area weights for the MutlivariatekCRPS score computation.
            feature_weights: Tensor of feature weights for the MutlivariatekCRPS score computation.
            fair: Calculate a "fair" (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1
            p_norm: p-norm to use for the MutlivariatekCRPS score computation. Defaults to 1.5.
            beta: beta parameter for the MutlivariatekCRPS score computation. Defaults to 1.0.
            implementation: Implementation of the MutlivariatekCRPS score computation. Defaults to "vectorized".
            ignore_nans: Ignore nans in the loss and apply methods ignoring nans for measuring the loss. Defaults to False.
        """
        super().__init__()

        assert group_on_dim in [-3, -2, -1], f"Invalid group dimension: {group_on_dim}"

        self.fair = fair
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum
        self.group_on_dim = group_on_dim
        self.implementation = implementation


        # Node weights normalization
        self.register_buffer("node_weights", node_weights[..., None], persistent=False)    
        self.register_buffer(
            "node_weights_normalized",
            self.node_weights / torch.sum(self.node_weights),
            persistent=False,
        )

        if self.group_on_dim == -2:
            self.node_weights_normalized = torch.pow(self.node_weights_normalized, 1 / beta)

        # Feature weights normalization
        assert feature_weights is not None, "Feature weights must be provided."
        self.register_buffer("feature_weights", feature_weights, persistent=False)
        self.register_buffer(
            "feature_weights_normalized",
            self.feature_weights / self.feature_weights.numel(),
            persistent=False,
        )

        if self.group_on_dim == -1:
            self.feature_weights_normalized = torch.pow(self.feature_weights_normalized, 1 / beta)
            self.node_weights_normalized = self.node_weights_normalized[..., 0]

        # beta and p_norm initialization
        self.register_buffer("beta", torch.as_tensor(beta), persistent=False)
        if self.beta.ndim == 0:
            self.beta = self.beta.unsqueeze(0)

        self.p_norm = p_norm

        # Dictionary of forward functions based on group dimension
        self.forward_funcs = {
            -1: self._forward_feature,
            -2: self._forward_spatial,
            -3: self._forward_temporal,
        }

        self._kernel_crps_impl = {"low_mem": self._kernel_crps_low_mem, "vectorized": self._kernel_crps_vectorized}


    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._kernel_crps_impl[self.implementation](preds, targets)

    def _kernel_crps_low_mem(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multivariate Kernel CRPS with low memory usage.

        This method computes the Kernel Continuous Ranked Probability Score (kCRPS) in a low-memory 
        manner by iterating over ensemble members and calculating pairwise multivariate norms.

        Args:
            preds (torch.Tensor): Predicted ensemble, shape (batch_size, ens_input, timesteps, latlon, n_vars)
            targets (torch.Tensor): Ground truth, shape (batch_size, ens_target, timesteps, latlon, n_vars)

        Returns:
            torch.Tensor: The point-wise multivariate CRPS, shape (batch_size, timesteps, latlon)
        """
        ens_input = preds.shape[1]
        ens_target = targets.shape[1]

        # Rearrange preds to shape (batch_size, timesteps, latlon, n_vars, ens_size) 
        preds = einops.rearrange(preds, "b e t l n -> b t l n e")
        targets = targets  # Shape remains (batch_size, timesteps, latlon, n_vars)

        if ens_input == 0:
            raise ValueError("Ensemble size must be greater than 0.")

        # Compute error: mean of multivariate norms between targets and each ensemble member
        # Expand targets to have an ensemble dimension for broadcasting
        diffs = targets - preds  # Shape: (b, t, l, n, e)
        error = torch.mean(torch.linalg.norm(diffs, ord=self.p_norm, dim=self.group_on_dim), dim=-1)  # Shape: (b, t, l)

        if ens_input == 1:
            return error  # No ensemble variance to compute

        # Compute ensemble variance: sum of pairwise multivariate norms between ensemble members
        coef = -1.0 / (ens_input * (ens_input - 1)) if self.fair else -1.0 / (ens_input ** 2)
        ens_var = torch.zeros_like(error)  # Initialize ensemble variance tensor

        for i in range(ens_input):
            # Compute differences between the i-th ensemble member and all following members
            pair_diffs = preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]  # Shape: (b, t, l, n, e_i)
            # Compute multivariate norms of pairwise differences
            pair_norms = torch.linalg.norm(pair_diffs, ord=self.p_norm, dim=self.group_on_dim)  # Shape: (b, t, l, e_i)
            # Sum over the ensemble members to accumulate ensemble variance
            ens_var += pair_norms.sum(dim=-1)  # Shape: (b, t, l)

        # Scale the ensemble variance
        ens_var = coef * ens_var  # Shape: (b, t, l)
        
        if ens_target != 1:
            coef = -1.0 / (ens_target * (ens_target - 1)) if self.fair else -1.0 / (ens_target ** 2)
            # Compute differences between the i-th ensemble member and all following members
            pair_diffs = targets[..., i].unsqueeze(-1) - targets[..., i + 1 :]  # Shape: (b, t, l, n, e_i)
            # Compute multivariate norms of pairwise differences
            pair_norms = torch.linalg.norm(pair_diffs, ord=self.p_norm, dim=self.group_on_dim)  # Shape: (b, t, l, e_i)
            # Sum over the ensemble members to accumulate ensemble variance
            ens_var += (pair_norms.sum(dim=-1) * coef)  # Shape: (b, t, l)

            
        # Return the final CRPS
        return error + ens_var  # Shape: (b, t, l)
    
    def _kernel_crps_vectorized(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Kernel (ensemble) CRPS considering both variables and ensemble members.
        # Note this implementation is generalized to the case where y is not a single vector but an ensemble of vectors

        """
        ens_input = preds.shape[1]
        ens_target = targets.shape[1]

        
        # old way to compute error which is only valid when y is a single vector
        # error = torch.mean(
        #     torch.linalg.norm(targets - preds, ord=self.p_norm, dim=self.group_on_dim).pow(self.beta), 
        #     dim=1)  # (batch_size, timesteps, latlon)

        # new way to compute error which is valid when y is an ensemble of vectors
        error = torch.mean(
            torch.linalg.norm(targets.unsqueeze(1) - preds.unsqueeze(2), ord=self.p_norm, dim=self.group_on_dim).pow(self.beta), 
            dim=(1, 2))  # (batch_size, timesteps, latlon)

        # If only one ensemble member, return error (no spread term)
        if ens_input == 1:
            return error

        # Now calculate ensemble variance (spread) based on the vector norms across ensemble members
        coef = (-0.5 / (ens_input * (ens_input - 1)) ) if self.fair else (-0.5 / (ens_input**2))
        pairwise_diffs = torch.linalg.norm(preds.unsqueeze(1) - preds.unsqueeze(2), ord=self.p_norm, dim=self.group_on_dim).pow(self.beta)
        variation = pairwise_diffs.sum(dim=(1, 2)) * coef

        if ens_target != 1:
            coef = (-0.5 / (ens_target * (ens_target - 1)) ) if self.fair else (-0.5 / (ens_target**2))
            pairwise_diffs = torch.linalg.norm(targets.unsqueeze(1) - targets.unsqueeze(2), ord=self.p_norm, dim=self.group_on_dim).pow(self.beta)
            variation += (pairwise_diffs.sum(dim=(1, 2)) * coef)
            
        return error + variation

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for MutlivariatekCRPS score calculation.
        
        Args:
            preds: predicted ensemble, shape (batch_size, nens_input, timesteps, latlon, n_vars)
            target: ground truth, shape (batch_size, nens_target, timesteps, latlon, n_vars)
            squash: bool, optional
                Reduce the spatial and feature dimensions
            feature_indices: indices of the features to scale the loss by
            feature_scale: bool, optional
                Scale the loss by the feature weights
        """
        return self.forward_funcs[self.group_on_dim](preds, target, squash, feature_indices, feature_scale)

    def _forward_spatial(self, preds: torch.Tensor, target: torch.Tensor, squash: bool, feature_indices: torch.Tensor | None, feature_scale: bool) -> torch.Tensor:
        """Forward pass for spatial grouping (group_on_dim = -2)."""
        preds = preds * self.node_weights_normalized
        target = target * self.node_weights_normalized

        kcrps = self._kernel_crps(preds, target)

        if feature_scale:
            kcrps = kcrps * self._scale_feature(feature_indices)

        return self._reduce_output(kcrps, squash)

    def _forward_feature(self, preds: torch.Tensor, target: torch.Tensor, squash: bool, feature_indices: torch.Tensor | None, feature_scale: bool) -> torch.Tensor:
        """Forward pass for feature grouping (group_on_dim = -1)."""
        if feature_scale:
            feature_scale = self._scale_feature( feature_indices)
            preds, target = preds * feature_scale, target * feature_scale

        kcrps = self._kernel_crps(preds, target)

        kcrps = kcrps * self.node_weights_normalized

        return self._reduce_output(kcrps, squash)

    def _forward_temporal(self, preds: torch.Tensor, target: torch.Tensor, squash: bool, feature_indices: torch.Tensor | None, feature_scale: bool) -> torch.Tensor:
        """Forward pass for temporal grouping (group_on_dim = -3)."""
        # No node weights applied since this is temporal
        kcrps = self._kernel_crps(preds, target)

        kcrps = kcrps * self.node_weights_normalized
        
        if feature_scale:
            kcrps = kcrps * self._scale_feature(feature_indices)

        return self._reduce_output(kcrps, squash)

    def _scale_feature(self, feature_indices: torch.Tensor | None) -> torch.Tensor:
        """Apply scaling for feature dimensions."""
        return self.feature_weights_normalized if feature_indices is None else self.feature_weights_normalized[..., feature_indices]

    #TODO (Rilwan-ade): this _squash function should be moved to a general Loss class
    def _reduce_output(self, kcrps: torch.Tensor, squash: bool) -> torch.Tensor:
        """Reduce the output to a single value or return the tensor."""
        if squash:
            kcrps = kcrps.sum(dim=(-2, -1))

        return kcrps.mean(0)

    def name(self) -> str:
        """Generate a log name based on parameters."""
        fair_str = "fair" if self.fair else ""

        kernel_str  = f"pnorm_{self.p_norm}"

        return f"{fair_str}kcrps_{kernel_str}_b{self.beta.item()}"

class GroupedMultivariatekCRPS(MultivariatekCRPS):
    """Grouped MutlivariatekCRPS Score 
        Grouping strategies:
            - Spatial: kCRPS patches
            - Feature: Grouping by variable, Grouping by pressure level
            - Temporal: ??

    
    It includes
    efficiency improvements through operation batching 

    Attributes
    ----------
        options(List[str]): List of available kCRPS patch options.
        node_weights(Tensor): Tensor of area weights for the MutlivariatekCRPS score computation.

        beta(Optional[Union[Tensor, float]]): beta parameter for the MutlivariatekCRPS score computation. Defaults to 1.0.False.

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
            Initializes the GroupedMutlivariatekCRPSScore_voronoi instance.

        forward(self, preds: Tensor, target: Tensor, squash: bool = True, enum_feature_weight_scalenorm: int = 0,
                enum_area_weight_scalenorm: int = 0, indices_adjusted=None) -> Tensor:
            Forward pass of the Grouped MutlivariatekCRPS Score computation.

        name(self) -> str:
            Cached property to generate a log name based on the beta and group dimension.
    """

    _options_spatial_patch_method = ("voronoi_O96_h0", "voronoi_O96_h1", "voronoi_O96_h2", "voronoi_O96_h3")
    _options_feature_patch_method = ("group_by_variable", "group_by_pressurelevel")

    _options_short_name = types.MappingProxyType(
        {
            "voronoi_O96_h0": "vO96h0",
            "voronoi_O96_h1": "vO96h1",
            "voronoi_O96_h2": "v096h2",
            "voronoi_O96_h3": "vO96h3",
            "group_by_variable": "gvar",
            "group_by_pressurelevel": "gpl",
        },
    )

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor,
        patch_method: str,
        p_norm: float = 1.5,
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
        super().__init__(
            node_weights,
            feature_weights,
            beta=beta,
            p_norm=p_norm,
            group_on_dim=group_on_dim,
            fair=fair,
            logging=logging,
            implementation=implementation,
        )

        self.patch_method = patch_method

        self.li_patch_set = li_patch_set or self._get_patch_sets(
            patch_method,
            data_indices_model_output,
            patches_dir=patches_dir,
        )
        # Check each patch set spans all the model output variables
        for patch_set in self.li_patch_set:
            self.check_patchset_for_unused_idxs(patch_set, data_indices_model_output.name_to_index)

        self.num_patch_sets = len(self.li_patch_set)

        self.li_patch_set_len = [torch.as_tensor(len(patch_set)) for patch_set in self.li_patch_set]

        self.op_batching = op_batching
        self.li_patch_set_opbatched = None
        if self.op_batching > 1:
            self.li_patch_set_opbatched = self.stack_patches(self.li_patch_set)
            self.num_patch_sets_opbatched = len(self.li_patch_set_opbatched)

        LOGGER.debug("Number of Patch Sets: %d", self.num_patch_sets)
        # LOGGER.info(f"\nPatch Sets: {str(self.li_patch_set)}")

    @staticmethod
    def check_patchset_for_unused_idxs(patch_set: dict[str, list[Tensor.int]], full_map_name_idx: dict[str, int]) -> None:
        # Does in-place modification of patch_set
        # Checks to see if each model output variable is in any of the patch sets
        all_idxs_accross_patches = set()
        for idxs in patch_set.values():
            all_idxs_accross_patches.update(idxs.to("cpu").numpy())

        idxs_not_in_patch_set = set(full_map_name_idx.values()) - all_idxs_accross_patches
        if len(idxs_not_in_patch_set) > 0:
            names = [k for k, v in full_map_name_idx.items() if v in idxs_not_in_patch_set]
            msg = f"Model output variables {names} not found in any patch set"
            raise ValueError(msg)

    def _get_patch_sets(
        self,
        patch_method: str,
        data_indices_model_output: DictConfig | None = None,
        patches_dir: Path | str | None = None,
    ) -> dict[str, int]:
        """Get list of patche sets based on the patch method."""
        if self.group_on_dim == -1:
            assert (
                patch_method in self._options_feature_patch_method
            ), f"Patch method {patch_method} not recognized, options are {self._options_feature_patch_method}"

            assert data_indices_model_output is not None, "Model output data indices must be provided"

            if patch_method == "group_by_variable":
                # group_by_variable method only returns one patch set
                name_to_index: dict[str, int] = data_indices_model_output.name_to_index

                # Group the unique variable names across different pressure levels
                map_var_idxs = defaultdict(list)
                for var_feature_pl, idx in name_to_index.items():

                    # Handle variables of the form "var_pl"
                    if "_" in var_feature_pl and var_feature_pl.split("_")[-1].isdigit():
                        var, pl = var_feature_pl.split("_")
                        map_var_idxs[var].append(idx)

                    # Handling other variables which have some form of height in diff format e.g. 10v and 10u
                    else:
                        # remove the numbers from the var_name
                        var_fmtd = "".join([i for i in var_feature_pl if not i.isdigit()])
                        var_fmtd = var_fmtd.strip("_")
                        map_var_idxs[var_fmtd].append(idx)

                # patch_set = [torch.as_tensor(var_idxs) for var_idxs in map_var_idxs.values()]
                patch_set = {key: torch.as_tensor(val) for key, val in map_var_idxs.items()}

                li_patch_set = [patch_set]

            elif patch_method == "group_by_pressurelevel":
                # group surface variables togeter
                name_to_index: dict[str, int] = data_indices_model_output.name_to_index

                # Create groupings for each pressure level
                map_pl_idxs = defaultdict(list)

                for var_feature_pl, idx in name_to_index.items():

                    # Handle variables of the form "var_pl"
                    if "_" in var_feature_pl and var_feature_pl.split("_")[-1].isdigit():
                        var, pl = var_feature_pl.split("_")
                        map_pl_idxs[pl].append(idx)

                    else:
                        # remove the numbers from the var_name
                        pl_fmtd = "".join([i for i in var_feature_pl if i.isdigit()])

                        if pl_fmtd != "":
                            # if under 50 hPa, group together with surface variables
                            if int(pl_fmtd) < 50:
                                pl_fmtd = "sfc"
                            map_pl_idxs[pl_fmtd].append(idx)
                        else:
                            map_pl_idxs[var_feature_pl].append(idx)

                patch_set = {key: torch.as_tensor(val) for key, val in map_pl_idxs.items()}
                li_patch_set = [patch_set]

        elif self.group_on_dim == -2:
            assert (
                patch_method in self._options_spatial_patch_method
            ), f"Patch method {patch_method} not recognized, options are {self._options_spatial_patch_method}"

            try:
                # directory contains list of patch groups with name format ddd.npz
                # if random_patches is True, all patch group
                # if not then only the first patch group is selected
                patch_group_paths_regex = Path(patches_dir) / patch_method / "patches_data" / "*.npz"
                patch_group_files = Path.glob(patch_group_paths_regex)

                # patch_group_npz_files = [np.load(patch_file, fix_imports=False) for patch_file in patch_group_files]
                import multiprocessing as mp

                # Gets the number of available CPUs available to this pid
                num_cpus = len(os.sched_getaffinity(os.getpid()))
                with mp.Pool(num_cpus) as pool:
                    li_patch_set_list = pool.map(process_file, patch_group_files)
                li_patch_set = [
                    {f"patch_group_{i}": torch.as_tensor(patch_set) for i, patch_set in enumerate(patch_set_list)}
                    for patch_set_list in li_patch_set_list
                ]

            except FileNotFoundError as e:
                msg = f"Patches file not found - Ensure patches are in {patches_dir}"
                raise FileNotFoundError(msg) from e

        else:
            msg = f"Patch method {patch_method} not recognized"
            raise ValueError(msg)

        # Checking each patch set to check that each contains the full set of model output variables
        for idx in range(len(li_patch_set)):
            patch_set = li_patch_set[idx]

        return li_patch_set

    def stack_patches(self, li_patch_set: list[dict[str, list[int]]]) -> list[dict[str, list[int]]]:
        """Stack patches together for operation batching.

        Patches stacked by similar size with at most self.op_batching patches in each
        stack
        """
        li_patch_set_opbatched = []
        for patch_set_idxs in li_patch_set.values():
            patch_set_opbatched = {}

            # Get the unique sizes of the patches
            unique_sizes = torch.unique(torch.tensor([p.shape for p in patch_set_idxs]), dim=0)

            for size in unique_sizes:
                same_size_patches = [p_idxs for p_idxs in patch_set_idxs if p_idxs.numel() == size]

                for i in range(0, len(same_size_patches), self.op_batching):
                    # stack the patches together along dim 0
                    stacked_patches = torch.stack(same_size_patches[i : i + self.op_batching], dim=0)
                    patch_set_opbatched.append(stacked_patches)

                    patch_set_opbatched[f"patch_size_{size}_group_{i}"] = stacked_patches

            li_patch_set_opbatched.append(patch_set_opbatched)

        LOGGER.info(f"Operation batching enabled with max batch size {self.op_batching}")
        for i, (patch_set, patch_set_opbatched) in enumerate(zip(li_patch_set, li_patch_set_opbatched)):
            LOGGER.info(f"\tPatch Set {i} patch count reduced: {len(patch_set)} -> {len(patch_set_opbatched)}")

        return li_patch_set_opbatched

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        deterministic: bool = False,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for MutlivariatekCRPS score calculation.

        Args:
            preds: predicted ensemble, shape (batch_size, ens_size, timesteps, latlon, n_vars)
            target: ground truth, shape (batch_size, ens_size, timesteps, latlon, n_vars)
            squash: bool, optional
                Reduce the spatial and feature dimensions
            feature_indices: indices of the features to scale the loss by
            feature_scale: bool, optional
                Scale the loss by the feature weights
        """

        if self.op_batching == 1:
            ridx = 0 if deterministic else torch.randint(0, self.num_patch_sets, (1,))
            patch_set = self.li_patch_set[ridx]
        else:
            ridx = 0 if deterministic else torch.randint(0, self.num_patch_sets_opbatched, (1,))
            patch_set = self.li_patch_set_opbatched[ridx]

        return self.forward_funcs[self.group_on_dim](preds, target, patch_set, squash, feature_indices, feature_scale)

    def _forward_feature(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        patch_set: dict[str, torch.Tensor],
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculate the Grouped Multivariate Kernel CRPS score.

        Args:
            preds: Tensor of shape (batch_size, ens_size, timesteps, latlon, num_vars)
            target: Tensor of shape (batch_size, timesteps, latlon, num_vars)
            patch_set: dictionary of patch sets
            squash: if False, return a (latlon, num_vars) tensor with the individual loss contributions
                    if True, return the (scalar) total loss
        Returns:
            Tensor of shape (batch_size, num_patches)
        """
        # Logic: features scaled before operation, area scaled after operation
        patch_set_len = len(patch_set)
        patches_set_values = list(patch_set.values())

        total_kcrps = target.new_zeros(target.shape[:-1])

        if feature_scale:
            preds = (
                preds * self.feature_weights_normalized
                if feature_indices is None
                else preds * self.feature_weights_normalized[..., feature_indices]
            )
            target = (
                target * self.feature_weights_normalized
                if feature_indices is None
                else target * self.feature_weights_normalized[..., feature_indices]
            )

        # Looped Loss
        for idx in torch.arange(0, patch_set_len):

            kcrps = self._kernel_crps(
                preds[..., patches_set_values[idx]],
                target[..., patches_set_values[idx]],
            )
            total_kcrps = total_kcrps + kcrps

        total_kcrps = total_kcrps 

        total_kcrps = total_kcrps * self.node_weights_normalized  # shape (bs, latlon)

        return self._reduce_output(total_kcrps, squash)

    def _forward_spatial(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        patch_set: dict[str, torch.Tensor],
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the Grouped Spatial Score computation.

        Args:
            preds (torch.Tensor): Predicted ensemble, shape (batch_size, ens_size, lat, lon, n_vars).
            target (torch.Tensor): Target values, shape (batch_size, latlon, n_vars).
            patch_set (dict[str, torch.Tensor]): dictionary of patch sets.
            squash (bool): If False, return a (latlon, n_vars) tensor with the individual loss contributions;
                if True, return the (scalar) total loss.

        Returns
        -------
            torch.Tensor: Grouped Spatial score. Scalar if squash is True; otherwise, shape based on patch_method.
        """
        patch_set_len = len(patch_set)
        patches_set_values = list(patch_set.values())

        total_kcrps = target.new_zeros((target.shape[0], target.shape[2]))  # shape (bs, n_vars)

        # Logic: area scaled before operation, feature after operation
        preds = preds * self.node_weights_normalized
        target = target * self.node_weights_normalized

        for idx in torch.arange(0, patch_set_len):
            # patch subselects on latlon dimension
            patch = patches_set_values[idx]

            if self.op_batching > 1:
                stacked_preds = preds[..., patch, :].permute(2, 0, 1, 3, 4)  # (op_batch, bs, ens_size, patch_size, nvar)
                stacked_target = target[..., patch, :].transpose(0, 1)  # (op_batch, bs, patch_size, nvar)

                kcrps = self._kernel_crps(stacked_preds, stacked_target)  # shape (op_batch, bs, n_vars)
                kcrps = kcrps.sum(dim=0)  # shape (bs, n_vars)

            else:
                kcrps = self._kernel_crps(preds[..., patch, :], target[..., patch, :])  # shape (bs, n_vars)

            total_kcrps = total_kcrps + kcrps.sum(dim=0)

        total_kcrps = total_kcrps   # shape (bs, n_vars)

        if feature_scale:
            total_kcrps = (
                total_kcrps * self.feature_weights_normalized
                if feature_indices is None
                else total_kcrps * self.feature_weights_normalized[..., feature_indices]
            )

        return self._reduce_output(total_kcrps, squash)

    @cached_property
    def name(self) -> str:

        beta_str = "b" + format(self.beta.item(), ".2g") if torch.is_tensor(self.beta) else "b" + format(self.beta, ".2g")

        patch_method_name = self._options_short_name.get(self.patch_method, self.patch_method)

        f_str = "f" if self.fair else ""

        

        return f"{f_str}kcrps_{patch_method_name}_{beta_str}"

    def is_calc_permissable(self, preds: torch.Tensor, target: torch.Tensor) -> bool:
        """Check if the calculation is permissible."""
        # Prevent calc on a subset of the variables
        return preds.shape[-1] == self.feature_weights_normalized.shape[-1]

