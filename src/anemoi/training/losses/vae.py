from __future__ import annotations

import logging

import torch
from torch import nn
from typing import Optional
LOGGER = logging.getLogger(__name__)
from torch import Tensor
from typing import Union
from omegaconf import DictConfig
from hydra import instantiate

# TODO: Make a AnemoiLoss base class that all losses inherit from and contains the shared functionality

class VAELoss(nn.Module):
    """Variational Autoencoder loss, combining reconstruction loss and KL divergence
    loss."""

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

        self.register_buffer("node_weights", node_weights[...,None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)

        if isinstance(reconstruction_loss, dict) or isinstance(reconstruction_loss, DictConfig):

            self.reconstruction_loss = instantiate(
                reconstruction_loss,
                node_weights=node_weights,
                feature_weights=feature_weights,
                 ignore_nans=ignore_nans, 
                **kwargs,
            )

        if isinstance(divergence_loss, dict) or isinstance(divergence_loss, DictConfig):
            # A similar scaling must be used for the divergence loss as is used for the reconstruction
            # the latent area weights are already handled (when they are normalized)
            # the feature dimension should be scaled by the scaling factor denominator used in reconstruction loss
            latent_node_weights = kwargs.get('latent_node_weights', latent_node_weights)
            self.divergence_loss = instantiate(
                divergence_loss,
                node_weights=latent_node_weights,
                feature_weights=latent_node_weights.new_ones(latent_node_weights.numel()),
                ignore_nans=ignore_nans,
                **kwargs
            )

        self.divergence_loss_weight = self.register_buffer("divergence_loss_weight", self.divergence_loss_weight )
        

    def forward(self, preds: Tensor, target: Tensor, squash: bool = True, **kwargs) -> Tensor:
        x_rec = preds
        x_target = target

        # TODO: extend the divergence_loss to be able to handle multiple layers of latent space
        z_mu = kwargs.pop("z_mu")
        z_logvar = kwargs.pop("z_logvar")

        if squash is True:
            div_loss = self.divergence_loss(z_mu, z_logvar, squash=True, **kwargs)
            rec_loss = self.reconstruction_loss(x_rec, x_target, squash=True, **kwargs)

            return {
                self.log_name: rec_loss + self.divergence_loss_weight * div_loss,
                f"{self.reconstruction_loss.log_name}": rec_loss,
                f"{self.divergence_loss.log_name}": div_loss,
            }

        else:
            # In this branch we squash the combined loss, but leave the individual losses unsquashed

            div_loss = self.divergence_loss(z_mu, z_logvar, squash=False, **kwargs)
            rec_loss = self.reconstruction_loss(x_rec, x_target, squash=False, **kwargs)

            loss_squashed = (rec_loss.sum() + self.divergence_loss_weight * div_loss.sum()) / x_rec.shape[0]

            return {
                self.log_name: loss_squashed,
                f"{self.reconstruction_loss.log_name}": rec_loss,
                f"{self.divergence_loss.log_name}": div_loss,
            }

    @cached_property
    def log_name(self):
        
        str_ = "vae"
        str_ += f"_{self.reconstruction_loss.log_name}"
        str_ += f"_{self.divergence_loss.log_name}-{self.divergence_loss_weight:.2f}"

        return str_

