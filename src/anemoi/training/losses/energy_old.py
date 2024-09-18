import os
import types
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch import nn
from anemoi.training.losses.utils import process_file
import logging
LOGGER = logging.getLogger(__name__)


class EnergyScore(nn.Module):
    """EnergyScore.

    Attributes
    ----------
    node_weights : torch.Tensor
        Weights for different areas in the loss calculation.
    feature_weights : Optional[torch.Tensor]
        Scaling factors applied to the loss.
    group_on_dim : int
        Dimension to group on for the loss calculation.
    power : torch.Tensor | int
        Power exponent for the energy loss.
    fair : bool
        Determines whether to apply the fair version of the score.
    ignore_nans : Optional[bool]
        Flag to determine whether to ignore NaN values.
    logging : str
        Logging level for the class.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: Optional[torch.Tensor],
        group_on_dim: int = -1,
        power: torch.Tensor | int = 1.0,
        p_norm: torch.Tensor | int = 2,
        fair: bool = True,
        ignore_nans: Optional[bool] = False,
        logging: str = "INFO",
        **kwargs,
    ) -> None:
        # TODO: evaluate version of this model with non-scaled output loss e.g. no feature_weights
        super().__init__()
        LOGGER.setLevel(logging)

        assert group_on_dim in [-2, -1], f"Grouping dimension {group_on_dim} not recognized"

        self.group_on_dim = group_on_dim
        self.fair = fair

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        area_weight_preoperation_exponent = power if self.group_on_dim == -2 else None
        feature_weight_scale_preoperation_exponent = power if self.group_on_dim == -1 else None

        self.register_buffer("node_weights", node_weights[..., None], persistent=False)
        self.register_buffer(
            "node_weights_scale_and_norm_constant",
            self.node_weights / torch.sum(self.node_weights),
            persistent=False,
        )

        if area_weight_preoperation_exponent is not None:
            self.node_weights_scale_and_norm_constant = torch.pow(
                self.node_weights_scale_and_norm_constant,
                1 / area_weight_preoperation_exponent,
            )

        # Setting up scaling constants for feature_weight_scales
        assert feature_weights is not None, "Loss scaling must be provided"
        self.register_buffer("feature_weight_scales", feature_weights, persistent=False)
        self.register_buffer(
            "feature_weight_scale_norm_constant",
            self.feature_weight_scales,
            persistent=False,
        )
        if feature_weight_scale_preoperation_exponent is not None:
            self.feature_weight_scale_norm_constant = torch.pow(
                self.feature_weight_scale_norm_constant,
                1 / feature_weight_scale_preoperation_exponent,
            )

        self.register_buffer("power", torch.as_tensor(power), persistent=False)
        if self.power.ndim == 0:
            self.power = self.power.unsqueeze(0)
        
        self.p_norm = p_norm

        if self.group_on_dim == -1:
            self.node_weights_scale_and_norm_constant = self.node_weights_scale_and_norm_constant[..., 0]

        self.forward_funcs = {
            -1: self._forward_feature,
            -2: self._forward_spatial,
        }

    def _calc_energy_score(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Ccalc energy score.

        Args:
            preds(torch.Tensor): Forecast realizations, shape(bs, nens, latlon, nvar).
            target(torch.Tensor): Ground truth("observations"), shape(bs, latlon, nvar).

        Returns
        -------
            torch.Tensor: The energy score loss component for gradient descent update,
            shape(bs, latlon / nvar).
        """
        pairwise_diff_predictor = torch.linalg.vector_norm(preds - target.unsqueeze(-3), dim=self.group_on_dim, ord=self.p_norm).pow(
            self.power,
        )
        precision_score = pairwise_diff_predictor.mean(dim=-2)  # shape (bs, latlon/nvar)

        pairwise_diff_copredictors = torch.linalg.vector_norm(
            preds.unsqueeze(-3) - preds.unsqueeze(-4),
            dim=self.group_on_dim,
            ord=self.p_norm,
        ).pow(self.power)
        # shape (bs, nens, nens, latlon/nvar)
        if self.fair:
            m = preds.shape[1]
            spread_score = pairwise_diff_copredictors.sum(dim=(-3, -2)) / (2 * m * (m - 1))
        else:
            spread_score = pairwise_diff_copredictors.mean(dim=(-3, -2)) * 0.5  # shape (bs, latlon/nvar)

        energy_score = precision_score - spread_score

        return energy_score  # shape ( bs, latlon/nvar )

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the loss function.

        Args:
            preds(Tensor): The predicted values. Shape: (batch_size, latlon, num_vars)
            target(Tensor): The target values. Shape: (batch_size, latlon, num_vars)
            squash(bool, optional): .
            feature_indices(Tensor, optional): .

        Returns
        -------
           torch.Tensor, : The energy score. Shape: (1) if squash is True, otherwise(latlon, num_vars)
        """
        return self.forward_funcs[self.group_on_dim](preds, target, squash, feature_indices, feature_scale)

    def _forward_spatial(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the loss function.

        Args:
            preds(Tensor): The predicted values. Shape: (batch_size, latlon, num_vars)
            target(Tensor): The target values. Shape: (batch_size, latlon, num_vars)
            squash(bool, optional): .

        Returns
        -------
           torch.Tensor, : The energy score. Shape: (1) if squash is True, otherwise(latlon, num_vars)
        """
        # Calculate the feature_weight_scales which when applied prior to the energy score calculation
        # results in the effect feature_weight_scales at the end
        # In other loss functions we apply the feature_weight_scales after the operation

        # Logic: areas scaled before operation, feature after operation

        preds = preds * self.node_weights_scale_and_norm_constant
        target = target * self.node_weights_scale_and_norm_constant

        energy_score = self._calc_energy_score(preds, target)  # shape (bs, nvar)

        if feature_scale:
            energy_score = (
                energy_score * self.feature_weight_scale_norm_constant
                if feature_indices is None
                else energy_score * self.feature_weight_scale_norm_constant[..., feature_indices]
            )

        if squash:
            return energy_score.sum() / energy_score.shape[0]

        return energy_score.sum(dim=0) / energy_score.shape[0]

    def _forward_feature(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the loss function.

        Args:
            preds(Tensor): The predicted values. Shape: (batch_size, latlon, num_vars)
            target(Tensor): The target values. Shape: (batch_size, latlon, num_vars)
            squash(bool, optional): .

        Returns
        -------
           torch.Tensor, : The energy score. Shape: (1) if squash is True, otherwise(latlon, num_vars)
        """
        # Logic: features scaled before operation, area scaled after operation

        if feature_scale:
            preds = (
                preds * self.feature_weight_scale_norm_constant
                if feature_indices is None
                else preds * self.feature_weight_scale_norm_constant[..., feature_indices]
            )
            target = (
                target * self.feature_weight_scale_norm_constant
                if feature_indices is None
                else target * self.feature_weight_scale_norm_constant[..., feature_indices]
            )

        energy_score = self._calc_energy_score(preds, target)  # shape (bs, latlon)
        energy_score = energy_score * self.node_weights_scale_and_norm_constant  # shape (bs, latlon)

        if squash:
            return energy_score.sum() / (energy_score.shape[0])  #
        return energy_score.sum(dim=0) / energy_score.shape[0]  # shape (latlon)

    @cached_property
    def log_name(self):
        # power_str = "p" + str(self.power.item()) if torch.is_tensor(self.power) else "p" + str(self.power)
        power_str = "p" + format(self.power.item(), ".2g") if torch.is_tensor(self.power) else "p" + format(self.power, ".2g")
        dim_str = f"d{self.group_on_dim}"

        # fair / not fair
        f_str = "f" if self.fair else ""

        # pnorm str
        pnorm_str = f"_pnorm{self.p_norm}" if self.p_norm != 2 else ""

        name = f"{f_str}energy_{dim_str}_{power_str}{pnorm_str}"

        return name


class PatchedEnergyScore(EnergyScore):
    """Patched Variogram Score using Voronoi patches.

    This class computes the variogram score over spatial patches defined by Voronoi regions. It includes
    efficiency improvements through operation batching and the use of CUDA streams.

    Attributes
    ----------
        options(List[str]): List of available Voronoi patch options.
        node_weights(Tensor): Tensor of area weights for the variogram score computation.

        power(Optional[Union[Tensor, float]]): Power parameter for the variogram score computation. Defaults to 1.0.False.

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
                 power: Optional[Union[Tensor, float]] = 1.0, random_patches: bool = False,
                 cuda_stream_count: int = 1, op_batching: int = 1, **kwargs) -> None:
            Initializes the PatchedVariogramScore_voronoi instance.

        forward(self, preds: Tensor, target: Tensor, squash: bool = True, enum_feature_weight_scalenorm: int = 0,
                enum_area_weight_scalenorm: int = 0, indices_adjusted=None) -> Tensor:
            Forward pass of the Patched Variogram Score computation.

        log_name(self) -> str:
            Cached property to generate a log name based on the power and group dimension.
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
        power: Optional[float] = 1.0,
        group_on_dim: int = -1,
        fair: bool = True,
        logging: str = "INFO",
        data_indices_model_output: Optional[DictConfig] = None,
        op_batching: int = 1,
        li_patch_set: Optional[list[dict[str, list[int]]]] = None,
        patches_dir: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            node_weights,
            feature_weights,
            power=power,
            group_on_dim=group_on_dim,
            fair=fair,
            logging=logging,
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
            raise ValueError(f"Model output variables {names} not found in any patch set")

    def _get_patch_sets(
        self,
        patch_method: str,
        data_indices_model_output: Optional[DictConfig] = None,
        patches_dir: Optional[Union[Path, str]] = None,
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

                    # # Handle surface variables of form "sfc_var"
                    # elif var_feature_pl[:3] == "sfc":
                    #     var_fmtd = var_feature_pl[3:]
                    #     var_fmtd = var_fmtd.strip("_")
                    #     map_var_idxs[var_fmtd].append(idx)

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

                    # # Group surface variables together
                    # elif var_feature_pl[:3] == "sfc":
                    #     map_pl_idxs["sfc"].append(idx)

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
                raise FileNotFoundError(f"Patches file not found - Ensure patches are in {patches_dir}") from e

        else:
            raise ValueError(f"Patch method {patch_method} not recognized")

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
        feature_indices: Optional[torch.Tensor] = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:

        if self.op_batching is not None:
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
        feature_indices: Optional[torch.Tensor] = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculate the patched energy score.

        Args:
            preds: Tensor of shape (batch_size, ens_size, latlon, num_vars)
            target: Tensor of shape (batch_size, latlon, num_vars)
            patch_set: dictionary of patch sets
            squash: if False, return a (latlon, num_vars) tensor with the individual loss contributions
                    if True, return the (scalar) total loss
        Returns:
            Tensor of shape (batch_size, num_patches)
        """
        # Logic: features scaled before operation, area scaled after operation
        patch_set_len = len(patch_set)
        patches_set_values = list(patch_set.values())

        total_es = target.new_zeros(target.shape[:2])

        if feature_scale:
            preds = (
                preds * self.feature_weight_scale_norm_constant
                if feature_indices is None
                else preds * self.feature_weight_scale_norm_constant[..., feature_indices]
            )
            target = (
                target * self.feature_weight_scale_norm_constant
                if feature_indices is None
                else target * self.feature_weight_scale_norm_constant[..., feature_indices]
            )

        # Looped Loss
        for idx in torch.arange(0, patch_set_len):

            es = self._calc_energy_score(
                preds[..., patches_set_values[idx]],
                target[..., patches_set_values[idx]],
            )
            total_es = total_es + es

        total_es = total_es / patch_set_len

        total_es = total_es * self.node_weights_scale_and_norm_constant  # shape (bs, latlon)

        if squash:
            return total_es.sum() / total_es.shape[0]

        return total_es.sum(axis=0) / total_es.shape[0]  # shape (latlon)

    def _forward_spatial(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        patch_set: dict[str, torch.Tensor],
        squash: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the Patched Spatial Score computation.

        Args:
            preds (torch.Tensor): Predicted ensemble, shape (batch_size, ens_size, lat, lon, n_vars).
            target (torch.Tensor): Target values, shape (batch_size, latlon, n_vars).
            patch_set (dict[str, torch.Tensor]): dictionary of patch sets.
            squash (bool): If False, return a (latlon, n_vars) tensor with the individual loss contributions;
                if True, return the (scalar) total loss.

        Returns
        -------
            torch.Tensor: Patched Spatial score. Scalar if squash is True; otherwise, shape based on patch_method.
        """
        patch_set_len = len(patch_set)
        patches_set_values = list(patch_set.values())

        total_es = target.new_zeros((target.shape[0], target.shape[2]))  # shape (bs, n_vars)

        # Logic: area scaled before operation, feature after operation
        preds = preds * self.node_weights_scale_and_norm_constant
        target = target * self.node_weights_scale_and_norm_constant

        for idx in torch.arange(0, patch_set_len):
            # patch subselects on latlon dimension
            patch = patches_set_values[idx]

            if self.op_batching > 1:
                stacked_preds = preds[..., patch, :].permute(2, 0, 1, 3, 4)  # (op_batch, bs, ens_size, patch_size, nvar)
                stacked_target = target[..., patch, :].transpose(0, 1)  # (op_batch, bs, patch_size, nvar)

                es = self._calc_energy_score(stacked_preds, stacked_target)  # shape (op_batch, bs, n_vars)
                es = es.sum(dim=0)  # shape (bs, n_vars)

            else:
                es = self._calc_energy_score(preds[..., patch, :], target[..., patch, :])  # shape (bs, n_vars)

            total_es = total_es + es.sum(dim=0)

        total_es = total_es / patch_set_len  # shape (bs, n_vars)

        if feature_scale:
            total_es = (
                total_es * self.feature_weight_scale_norm_constant
                if feature_indices is None
                else total_es * self.feature_weight_scale_norm_constant[..., feature_indices]
            )

        if squash:
            return total_es.sum() / total_es.shape[0]  # shape(1)

        return total_es.sum(0) / preds.shape[0]  # shape (n_vars)

    @cached_property
    def log_name(self):

        power_str = "p" + format(self.power.item(), ".2g") if torch.is_tensor(self.power) else "p" + format(self.power, ".2g")

        patch_method_name = self._options_short_name.get(self.patch_method, self.patch_method)

        f_str = "f" if self.fair else ""

        # pnorm str
        pnorm_str = f"_pnorm{self.p_norm}" if self.p_norm != 2 else ""

        return f"{f_str}penergy_{patch_method_name}_{power_str}{pnorm_str}"

    def is_calc_permissable(self, preds, target):
        """Check if the calculation is permissible."""
        # Prevent calc on a subset of the variables
        if preds.shape[-1] != self.feature_weight_scale_norm_constant.shape[-1]:
            return False

        return True


