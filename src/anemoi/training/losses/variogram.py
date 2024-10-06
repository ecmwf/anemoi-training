from __future__ import annotations
from typing import Optional
from functools import cached_property

import torch
from torch import nn
import logging
LOGGER = logging.getLogger(__name__)
from .mixins import TargetEachEnsIndepMixin
from typing import Union
# TODO(rilwan-adewoyin): Extend this variogram score to allow variogram on temporal patches
# TODO(rilwan-adewoyin): Add support for a sliding window for the variogram score (kinda)
# TODO(rilwan-adewoyin): Add support for asymetric variogram score e.g. if the model under-predicts variance


class VariogramScore(TargetEachEnsIndepMixin, nn.Module):
    """

    https://arxiv.org/pdf/1910.07325
    https://arxiv.org/html/2407.00650v1
    https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml
    
    Variogram score for evaluating spatial forecasts.
    The variogram score is a measure of the difference between the predicted and target values.
    It is calculated by taking the average of the squared differences between the predictions and targets.
    The score is then averaged over the spatial dimensions.
    # add latex of formula for support of sphinx docs
    .. math::
        V(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2
    
    Nuanced Understanding:
        When group_on_dim is -1, -2, -3 (feature, spatial, temporal dimensions):
            - If you have ens size > one in pred, and ens size = 1 in target, then variogram score; preds_diff is the ensemble average of the intergroup differences. 

            - If you have ens_size_inp == ens_size_target, then you calculate a variogram score for each ensemble member then take the average of these variogram scores in the ensemble dimension. e.g there is a preds_diff and target_diff per ensemble member
        When group_on_dim is -4 (ensemble dimension):
            - pred ens size must be greater than one and target ens size must be greater than one
            - Then one score calculated per ensemble per timestep, latlon and feature

    """
    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: Optional[torch.Tensor] = None,
        group_on_dim: int = -2,
        beta: float = 1.0,
        ignore_nans: bool = False,
        target_each_ens_indep: bool = False,
    ) -> None:
        """Variogram score for evaluating spatial forecasts.

        Parameters
        ----------
        node_weights : torch.Tensor
            Weight of each node in the spatial domain
        feature_weights : Optional[torch.Tensor]
            Weights for each feature, by default None
        beta : float
            The beta to which differences are raised, typically 2 for squared differences
        ignore_nans : bool, optional
            Whether to ignore NaN values in the calculations, by default False
        #NOTE: lower points are more resistant to outliers, allow more spread. common values for betas is 0.5, 1.0, 2.0
        target_each_ens_indep : bool, optional
            This flag calculates Variogram score such that we are trying to learn the dynamics of each ensemble member independently as opposed to trying to learn an average behaviour across the ensemble. 
            Only applicaple when ens_size_inp == ens_size_target and when group_on_dim is not -4
            (TODO (rilwan-adewoyin): clean this naming up after testing)
        """
        super().__init__()
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum
        self.target_each_ens_indep = target_each_ens_indep
        if self.target_each_ens_indep:
            assert group_on_dim in [-1, -2, -3], "Target full ensemble is only applicable when group_on_dim is one of the last three dimensions, Please make sure you understand the nuance of this Variogram Score"

        self.group_on_dim = group_on_dim
        assert self.group_on_dim in [-1, -2, -3, -4], "Group on dim must be one of the last four dimensions"

        self.register_buffer("beta", torch.as_tensor(beta), persistent=False)
        if self.beta.ndim == 0:
            self.beta = self.beta.unsqueeze(0)

        assert feature_weights is not None, "Feature weights must be provided."
        self.register_buffer("feature_weights", feature_weights, persistent=False)
        self.register_buffer(
            "feature_weights_normalized",
            self.feature_weights / self.feature_weights.numel(),
            persistent=False,
        )

        # Node weights normalization
        self.register_buffer("node_weights", node_weights[..., None], persistent=False)    
        self.register_buffer(
            "node_weights_normalized",
            self.node_weights / torch.sum(self.node_weights),
            persistent=False,
        )
       
        if self.group_on_dim == -1:
            self.feature_weights_normalized = torch.pow(self.feature_weights_normalized, 1 / (beta*2) )
            self.node_weights_normalized = self.node_weights_normalized[..., 0]
        elif self.group_on_dim == -2:
            self.node_weights_normalized = torch.pow(self.node_weights_normalized, 1 / (beta*2) )


        # beta and p_norm initialization
        self.register_buffer("beta", torch.as_tensor(beta**2), persistent=False)
        if self.beta.ndim == 0:
            self.beta = self.beta.unsqueeze(0)

        # Dictionary of forward functions based on group dimension
        self.forward_funcs = {
            -1: self._forward_feature,
            -2: self._forward_spatial,
            -3: self._forward_temporal,
            -4: self._forward_ensemble,
        }


    def _calc_variogram_score(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the variogram score.

        Args:
            preds (torch.Tensor): Predicted ensewmble, shape (batch_size, ens_size_p, timesteps, latlon, n_vars)
            target (torch.Tensor): Target values, shape (batch_size, ens_size_t, timesteps, latlon, n_vars)
        """
        #TODO (rilwan-adewoyin): Extend to a version that allows for EDA in input
        #NOTE: If EDAs are used / or target also has shape (batch_size, ens_size, (timesteps), latlon, n_vars)
        # then the mean in ensemble dimension should happen after the target_diff is calculated
        # NOTE: (rilwan-adewoyin): Write a sliding window wrapper for temporal loss scores

        pred_ens_size = preds.shape[1]
        target_ens_size = target.shape[1] if target.ndim == preds.ndim else 1

        if self.group_on_dim == -4:
            assert pred_ens_size > 1 and target_ens_size > 1, "Ensemble sizes must be > 1 for group_on_dim=-4"
            assert pred_ens_size == target_ens_size, "Ensemble sizes must be equal"

            # preds_diff = torch.abs(preds.unsqueeze(1) - preds.unsqueeze(2)).pow(self.beta).sum(dim=(1, 2))
            # target_diff = torch.abs(target.unsqueeze(1) - target.unsqueeze(2)).pow(self.beta).sum(dim=(1, 2))

            preds_diff = torch.abs(preds.unsqueeze(1) - preds.unsqueeze(2)).pow(self.beta)
            target_diff = torch.abs(target.unsqueeze(1) - target.unsqueeze(2)).pow(self.beta)

            vario_score = target_diff - preds_diff  # Compute difference between predictions and targets        
            vario_score = vario_score.pow(2)  # Square the difference
            vario_score = vario_score.sum(dim=(1, 2))
        
        else:
            preds_diff = torch.abs(preds.unsqueeze(self.group_on_dim) - preds.unsqueeze(self.group_on_dim - 1)).pow(self.beta).mean(dim=1)
            target_diff = torch.abs(target.unsqueeze(self.group_on_dim) - target.unsqueeze(self.group_on_dim - 1)).pow(self.beta).mean(dim=1)
            vario_score = target_diff - preds_diff  # Compute difference between predictions and targets     
            vario_score = vario_score.pow(2)  # Square the difference   
            vario_score = vario_score.sum(dim=(self.group_on_dim-1, self.group_on_dim))
                
            
        # elif self.target_each_ens_indep:
        #     # Here we are trying to target every ensemble member in the target
        #     # Intuition: there is an prediction of ensemble size 1 for every ensemble member in the target

        #     assert pred_ens_size == target_ens_size, "Ensemble sizes must be equal"
        #     preds_diff = torch.abs(preds.unsqueeze(self.group_on_dim) - preds.unsqueeze(self.group_on_dim - 1)).pow(self.beta) 
        #     target_diff = torch.abs(target.unsqueeze(self.group_on_dim) - target.unsqueeze(self.group_on_dim - 1)).pow(self.beta)
        #     vario_score = preds_diff - target_diff  # Compute difference between predictions and targets        
            
        #     vario_score = vario_score.pow(2)  # Square the difference
        #     vario_score = vario_score.sum(dim=(self.group_on_dim-1, self.group_on_dim))
        #     vario_score = vario_score.mean(dim=1)
        # Sum over the spatial dimensions (latlon)
        return vario_score

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: Union[bool, tuple] = True,
        feature_scale: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the Variogram Score computation.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, ens_size, (timesteps), latlon, n_vars)
        target : torch.Tensor
            Target values, shape (batch_size, (timesteps), latlon, n_vars)
        squash : bool, optional
            Whether to aggregate scores into a single value, by default True
        feature_scale : bool, optional
            Scale the loss by the feature weights, by default True
        feature_indices: Optional[torch.Tensor]
            Indices of the features to scale the loss by

        Returns
        -------
        torch.Tensor
            Variogram score, scalar if squash is True; otherwise shape (latlon, n_vars)
        """

        vario_score = self.forward_funcs[self.group_on_dim](preds, target, feature_scale, feature_indices, squash)
        return vario_score

    def _forward_feature(self, preds: torch.Tensor, target: torch.Tensor, feature_scale: bool = True, feature_indices: Optional[torch.Tensor] = None, squash: Union[bool, tuple] = True) -> torch.Tensor:
        """Forward pass for feature dimension.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, ens_size, timesteps, latlon, n_vars)
        target : torch.Tensor
            Target values, shape (batch_size, ens_size timesteps, latlon, n_vars)
        """             

        # Scale in the feature dimension
        if feature_scale:
            # vario_score = (vario_score * self.feature_weights/self.feature_weights.numel()) if feature_indices is None else (vario_score * self.feature_weights[..., feature_indices]/self.feature_weights.numel()  )

            preds = preds * self.feature_weights_normalized if feature_indices is None else self.feature_weights_normalized[..., feature_indices]
            target = target * self.feature_weights_normalized if feature_indices is None else self.feature_weights_normalized[..., feature_indices]

        # Calculate the variogram score
        vario_score = self._calc_variogram_score(preds, target)  # (batch_size, timesteps, latlon)

        # Scale in the spatial dimension
        vario_score = vario_score * self.node_weights_normalized

        return self._reduce_output(vario_score, squash)

    def _forward_spatial(self, preds: torch.Tensor, target: torch.Tensor, feature_scale: bool = True, feature_indices: Optional[torch.Tensor] = None, squash: Union[bool, tuple] = True) -> torch.Tensor:
        """Forward pass for spatial dimension.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, ens_size, timesteps, latlon, n_vars)
        target : torch.Tensor
            Target values, shape (batch_size, ens_size, timesteps, latlon, n_vars)
        feature_scale : bool, optional
            Scale the loss by the feature weights, by default True
        feature_indices: Optional[torch.Tensor]
            Indices of the features to scale the loss by
        squash : bool, optional
            Whether to aggregate scores into a single value, by default True

        Returns
        -------
        torch.Tensor
        """


        # Scale in the spatial dimension
        preds = preds * self.node_weights_normalized
        target = target * self.node_weights_normalized

        # Calculate the variogram score
        vario_score = self._calc_variogram_score(preds, target)  # (batch_size, timesteps, n_vars)

        # Scale in the feature dimension
        if feature_scale:
            vario_score = vario_score * self.feature_weights_normalized


        return self._reduce_output(vario_score, squash)

        # Scale in the spatial dimension
    
    def _forward_temporal(self, preds: torch.Tensor, target: torch.Tensor, feature_scale: bool = True, feature_indices: Optional[torch.Tensor] = None, squash: Union[bool, tuple] = True) -> torch.Tensor:
        """Forward pass for temporal dimension.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, ens_size, timesteps, latlon, n_vars)
        target : torch.Tensor
            Target values, shape (batch_size, ens_size, timesteps, latlon, n_vars)      
        """

        # Calculate the variogram score
        vario_score = self._calc_variogram_score(preds, target)  # (batch_size, latlon,n_vars) 

        # Scale in the spatial dimension
        vario_score = vario_score * self.node_weights_normalized

        # Scale in the feature dimension
        if feature_scale:
            vario_score = vario_score * self.feature_weights_normalized if feature_indices is None else self.feature_weights_normalized[..., feature_indices]

        return self._reduce_output(vario_score, squash)

    def _forward_ensemble(self, preds: torch.Tensor, target: torch.Tensor, feature_scale: bool = True, feature_indices: Optional[torch.Tensor] = None, squash: Union[bool, tuple] = True) -> torch.Tensor:
        
        # Calculate the variogram score
        vario_score = self._calc_variogram_score(preds, target)  # (batch_size, timesteps, latlon, n_vars) 

        # Scale in the spatial dimension
        vario_score = vario_score * self.node_weights_normalized
            
        # Scale in the feature dimension
        if feature_scale:
            vario_score = vario_score * self.feature_weights_normalized if feature_indices is None else self.feature_weights_normalized[..., feature_indices]

        return self._reduce_output(vario_score, squash)

    def _reduce_output(self, vario_score: torch.Tensor, squash: Union[bool, tuple] = True) -> torch.Tensor:
        """Reduce the output to a scalar."""
        if squash:
            if self.group_on_dim == -4:
                vario_score = self.sum_function(vario_score, dim=squash if isinstance(squash, tuple) else (-3, -2, -1))
            else:
                dim = tuple( (dim if dim>self.group_on_dim else dim+1) for dim in dim if dim != self.group_on_dim ) if isinstance(squash, tuple) else (-2, -1)
                vario_score = self.sum_function(vario_score, dim=dim)   
            
        return self.avg_function(vario_score, axis=0)

    @cached_property
    def name(self) -> str:
        return f"vgram_b{self.beta.item()}"
