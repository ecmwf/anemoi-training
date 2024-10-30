from __future__ import annotations

import logging

import torch
from torch import nn

from torch import Tensor
from omegaconf import DictConfig
from hydra.utils import instantiate
from functools import cached_property
# TODO: Make a AnemoiLoss base class that all losses inherit from and contains the shared functionality
LOGGER = logging.getLogger(__name__)
from typing import Optional

class VAELoss(nn.Module):
    """Variational Autoencoder loss, combining reconstruction loss and KL divergence
    loss.
    """

    def __init__(
        self,
        node_weights: Tensor,
        feature_weights: Tensor,
        reconstruction_loss: dict | DictConfig | nn.Module,
        divergence_loss:  dict | DictConfig | nn.Module,
        latent_node_weights: Tensor,
        divergence_loss_weight: Tensor | float = 1e-2,
        ignore_nans: bool | None = False,
        **kwargs,
    ):
        super().__init__()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        if isinstance(reconstruction_loss, (dict, DictConfig)):

            self.reconstruction_loss = instantiate(
                reconstruction_loss,
                node_weights=node_weights,
                feature_weights=feature_weights,
                 ignore_nans=ignore_nans,
                **kwargs,
            )
        elif isinstance(reconstruction_loss, nn.Module):
            self.reconstruction_loss = reconstruction_loss
        else:
            raise ValueError(f"Invalid reconstruction loss: {reconstruction_loss}")

        if isinstance(divergence_loss, (dict, DictConfig)):
            # A similar scaling must be used for the divergence loss as is used for the reconstruction
            # the latent area weights are already handled (when they are normalized)
            # the feature dimension should be scaled by the scaling factor denominator used in reconstruction loss
            
            self.divergence_loss = instantiate(
                divergence_loss,
                node_weights=latent_node_weights,
                ignore_nans=ignore_nans,
                **kwargs,
            )
        elif isinstance(divergence_loss, nn.Module):
            self.divergence_loss = divergence_loss
        else:
            raise ValueError(f"Invalid divergence loss: {divergence_loss}")

        self.register_buffer("divergence_loss_weight", torch.tensor(divergence_loss_weight))

    def forward(self, preds: Tensor, target: Tensor, squash: bool|tuple = True, feature_scale: bool = True, feature_indices: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Parameters
        ----------
        preds : torch.Tensor
            Predictions tensor, shape (bs, (timesteps), lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, (timesteps), lat*lon, n_outputs)
        squash : bool, optional
            Whether to squash the loss, by default True

        Returns
        -------
        torch.Tensor
            Weighted VAE loss.
        """
        x_rec = preds
        x_target = target

        # TODO: extend the divergence_loss to be able to handle multiple layers of latent space
        z_mu = kwargs.pop("z_mu")
        z_logvar = kwargs.pop("z_logvar")

        div_loss = self.divergence_loss(z_mu, z_logvar, squash=squash, feature_scale=False, feature_indices=feature_indices)
        rec_loss = self.reconstruction_loss(x_rec, x_target, squash=squash, feature_scale=feature_scale, feature_indices=feature_indices)

        # NOTE: The rec_loss and divergence loss can not consistently be broadcasted to have a similar shape
        vae_loss = (rec_loss.sum() + self.divergence_loss_weight * div_loss.sum())
        
        return {
            self.name: vae_loss,
            f"{self.reconstruction_loss.name}": rec_loss,
            f"{self.divergence_loss.name}": div_loss,
        }

    @cached_property
    def name(self):

        str_ = "vae"
        str_ += f"_{self.reconstruction_loss.name}"
        str_ += f"_{self.divergence_loss.name}-{self.divergence_loss_weight:.2f}"

        return str_