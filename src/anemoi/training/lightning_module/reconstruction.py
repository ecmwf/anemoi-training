from __future__ import annotations


import numpy as np
import torch

from torch import Tensor
from torch.utils.checkpoint import checkpoint
from anemoi.training.lightning_module.mixins import DeterministicCommunicationMixin, EnsembleCommunicationMixin
from anemoi.training.lightning_module.anemoi import AnemoiLightningModule
from anemoi.models.interface import AnemoiModelInterfaceReconstruction
from anemoi.training.data.inicond import EnsembleInitialConditions
from hydra.utils import instantiate


import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from collections.abc import Mapping
LOGGER = logging.getLogger(__name__)

#NOTE: (currently this only works for VAE)
# the z_mu, z_logvar wouldn't be in a VQ-VAE, generalize this later
class ReconstructionLightningModule(AnemoiLightningModule):
    def __init__(self, config, graph_data, statistics, statistics_tendencies, data_indices, metadata):
        super().__init__(config, graph_data, statistics, data_indices, metadata, model_cls=AnemoiModelInterfaceReconstruction)

        self.latlons_hidden = [graph_data[h_name].x for h_name in config.graph.hidden]
        self.latent_weights = graph_data[config.graph.hidden[-1]][config.model.node_loss_weight].squeeze()

    @staticmethod
    def get_latent_weights(graph_data, config):
        return graph_data[config.graph.hidden[-1]][config.model.node_loss_weight].squeeze()



    def _step(
        self,
        batch: Tensor, # shape (bs, multistep, latlon, nvars_full)
        batch_idx: int = 0,
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping, Tensor]:

        batch = self.model.pre_processor_state(batch, inplace=False)  # shape = (bs, nens_input, multistep, 

        x_inp = batch[..., self.data_indices.data.input.full]


        metrics = {}

        dict_model_outputs = self(x_inp) #  z_mu, li_z_mu, z_logvar
        x_rec = dict_model_outputs.pop("x_rec")
        # reconstructed state and VAE latent space descriptors (needed for KL loss)\

        # TODO (rilwan-ade): when multi level latent space is implemented, need to update
        # the calculate_val_metrics handle multiple levels??(think about this) (can't use skip connections)
        

        x_target = batch[
            ..., self.data_indices.data.output.full,
        ]  # shape = (bs, multistep, latlon, output.full)

        loss: Tensor | dict = checkpoint(self.loss, x_rec, x_target, **dict_model_outputs, squash=True, use_reentrant=False)

        if validation_mode:
            dict_tensors = self.get_proc_and_unproc_data(x_rec, x_target, x_inp)
            metrics_new = self.calculate_val_metrics(x_rec, x_target, **dict_model_outputs)

            metrics.update(metrics_new)

        # Reducing the time dimension out (leave it in)
        # TODO (rilwan-ade) make sure the z_logvar based evals handle the fact that there is a TIME dimension
        # TODO (rilwan-ade) make sure losses factor in this time and can report per time step in the sequence
        return loss, metrics, {**dict_tensors, **dict_model_outputs}

    def get_proc_and_unproc_data(self, x_rec: Tensor, x_target: Tensor, x_inp: Tensor):

        x_rec_postprocessed = self.model.post_processors(x_rec, in_place=False, data_index=self.data_indices.data.output.full)
        x_target_postprocessed = self.model.post_processors(x_target, in_place=False, data_index=self.data_indices.data.output.full)
        x_inp_postprocessed = self.model.post_processors(x_inp, in_place=False, data_index=self.data_indices.data.input.full)

        return {
            "x_rec": x_rec,
            "x_target": x_target,
            "x_inp": x_inp,
            "x_rec_postprocessed": x_rec_postprocessed,
            "x_target_postprocessed": x_target_postprocessed,
            "x_inp_postprocessed": x_inp_postprocessed,
        }

    def calculate_val_metrics(self, x_rec, x_target, x_inp, x_rec_postprocessed, x_target_postprocessed):
        metric_vals = {}
        timesteps = x_rec.shape[2]

        for metric in self.val_metrics:
            for mkey, indices in self.val_metric_ranges.items():
                # NOTE: feature_scaling is turned off here

                # for single metrics do no variable scaling and non processed data
                # TOOD (rilwan-ade): Update logging to use get_time_step and report lead time in hours
                if len(indices) == 1:
                    _args = (x_rec_postprocessed[..., indices], x_target_postprocessed[..., indices])
                else:
                    _args = (x_rec[..., indices], x_target[..., indices])
                
                m_value = metric(*_args, feature_scaling=False, squash=(-2, -1))

                if m_value.shape[0] == x_rec.shape[2]:
                    for i in range(timesteps):
                        metric_vals[f"{mkey}_{i + 1}"] = m_value[i]
                else:
                    metric_vals[f"{mkey}"] = m_value

        return metric_vals

class ReconstructionLightningModuleDeterministic(DeterministicCommunicationMixin, ReconstructionLightningModule):
    """Deterministic version of Reconstruction Lightning Module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ReconstructionLightningModuleICEnsemble(EnsembleCommunicationMixin, ReconstructionLightningModule):
    """Ensemble version of Reconstruction Lightning Module.
        When using initial condition pertubation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ensemble_ic_generator = EnsembleInitialConditions(config=self.config, data_indices=self.data_indices)


    def _step(
        self,
        batch: tuple[Tensor, Tensor],  # shape (bs, 1, multistep, latlon, nvars_full)
        batch_idx: int = 0,
        validation_mode: bool = False,
    ) -> tuple[Tensor, Mapping, Tensor]:
        
        """Run one  step.

        Args:
            batch: tuple
                Batch data..
                batch[0]: analysis, shape (bs, timesteps, nvar, latlon)
                batch[1] (optional): EDA perturbations, shape (timesteps, nens_input, nvar, latlon)
            batch_idx: int
                batch index   
            validation_mode: bool
        """
        x_center, x_ic = batch[0], batch[1]
        x_ic = self.ensemble_ic_generator(x_center, x_ic)  # shape (bs, nens_input, timesteps, latlon, nvars_full)

        batch_inp = self.model.pre_processor_state(x_ic, inplace=False)  # shape = (bs, nens_input, multistep, latlon, input.full)
        batch_target_center = self.model.pre_processor_state(x_center, inplace=False)  # shape = (bs, nens_target, multistep, latlon, 
        batch_target_ic = self.model.pre_processor_state(x_ic, inplace=False)  # shape = (bs, nens_target, multistep, latlon, 

        
        x_inp = batch_inp[..., self.data_indices.data.input.full]
        x_target_ic = batch_target_ic[..., self.data_indices.data.output.full] # shape = (bs,1, multistep, latlon, output.full)
        x_target_center = batch_target_center[..., self.data_indices.data.output.full] # shape = (bs,ens_ic, multistep, latlon, output.full)

        if self.config.training.target_ic:
            x_target = x_target_ic
        else:
            x_target = x_target_center

        metrics = {}

        dict_model_outputs = self(x_inp) #  z_mu, li_z_mu, z_logvar
        x_rec = dict_model_outputs.pop("x_rec")
        # reconstructed state and VAE latent space descriptors (needed for KL loss)\

        # TODO (rilwan-ade): when multi level latent space is implemented, need to update
        # the calculate_val_metrics handle multiple levels??(think about this) (can't use skip connections)
        
        loss: Tensor | dict = checkpoint(self.loss, x_rec, x_target, **dict_model_outputs, squash=True, use_reentrant=False)

        if validation_mode:
            dict_tensors = self.get_proc_and_unproc_data(x_rec, x_target, x_inp)
            metrics_new = self.calculate_val_metrics(x_rec, x_target, **dict_model_outputs)

            metrics.update(metrics_new)

        # Reducing the time dimension out (leave it in)
        return loss, metrics, {**dict_tensors, **dict_model_outputs}
