import torch
from torch import nn, Tensor
from typing import Optional

class IgnoranceScore(nn.Module):
    """Latitude-weighted Ignorance Score."""

    def __init__(
        self,
        node_weights: Tensor,
        feature_weights: Optional[Tensor] = None,
        eps: float = 1e-5,
        ignore_nans: bool = False,
    ) -> None:
        """Initialize Ignorance Score with latitude-weighted loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function.
        feature_weights : Optional[torch.Tensor of shape (N, )], optional
            Weight of each feature in the loss function, by default None.
        eps : float, optional
            A small epsilon to stabilize calculations, by default 1e-3.
        ignore_nans : bool, optional
            If True, NaNs will be ignored in the loss calculations, by default False.
        """
        super().__init__()
        
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights[...,None], persistent=True)
        self.register_buffer("eps", torch.as_tensor(eps), persistent=False)

        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights, persistent=True)
        else:
            self.feature_weights = None

    def _calc_ignorance_score(self, preds: Tensor, target: Tensor) -> Tensor:
        """Calculate the Ignorance Score."""
        var_preds = torch.var(preds, dim=1, correction=1) + self.eps  # Variance across ensemble dimension (bs, latlon, nvar)
        log_term = torch.log(2 * torch.pi * var_preds)  # Log term (bs, latlon, nvar)
        sum_term = (preds.mean(dim=1) - target).pow(2) / var_preds  # (bs, latlon, nvar)
        nll = 0.5 * (log_term + sum_term)  # Negative log-likelihood
        return nll  # (bs, latlon, nvar)

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        squash: bool = True,
        feature_scaling: bool = True,
        feature_indices: Optional[torch.Tensor] = None
    ) -> Tensor:
        """Forward pass for calculating Ignorance Score.

        Parameters
        ----------
        preds : torch.Tensor
            Predictions tensor, shape (bs, nens, timesteps, lat*lon, nvar).
        target : torch.Tensor
            Target tensor, shape (bs, timesteps, lat*lon, nvar).
        squash : bool, optional
            If True, the output will be squashed to a single value, by default True.
        feature_scaling : bool, optional
            If True, feature weights will be applied to the loss, by default True.
        feature_indices: indices of the features to scale the loss by

        Returns
        -------
        torch.Tensor
            Weighted Ignorance Score.
        """
        # Calculate Ignorance Score
        ignorance_score = self._calc_ignorance_score(preds, target)

        # Scale in feature dimension
        if feature_scaling:
            ignorance_score = ignorance_score * self.feature_weights if feature_indices is None else ignorance_score * self.feature_weights[..., feature_indices]
            ignorance_score = ignorance_score / self.feature_weights.numel()
            
            #Normalize by number of features

        # Scale in spatial dimension
        ignorance_score *= self.node_weights / self.sum_function(self.node_weights)

        # Squash: reduce the dimensions to a single value
        if squash:
            ignorance_score = self.sum_function(ignorance_score, dim=(1, 2))

        return ignorance_score.mean(dim=0) # (timesteps, latlon, nvar)

    @property
    def log_name(self) -> str:
        """Log name for the Ignorance Score."""
        return f"ignorance_eps{self.eps:.0e}"

