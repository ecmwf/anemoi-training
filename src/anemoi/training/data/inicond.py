import logging
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn

LOGGER = logging.getLogger(__name__)


class EnsembleInitialConditions(nn.Module):
    """Generates initial conditions for ensemble runs.

    Uses analysis and (optionally) EDA member data. This module has no buffers or
    trainable parameters.

    Important:
        The initial conditions passed to this module should already be recentered on the highest resolution signal available.
        Such that x_an is the highest resolution signal and x_ic is the same signal with EDA perturbations.
    """

    def __init__(self, config: DictConfig, data_indices: dict) -> None:
        """Initialise object.

        Parameters
        ----------
        data_indices : dict
            Indices of the training data
        """
        super().__init__()

        self.data_indices = data_indices
        self.multi_step = config.training.multistep_input

        self.nens_ic = config.training.ic_ensemble_size
        self.noise_sample_per_ic = config.training.noise_sample_per_ic
        self.nens_per_device = self.nens_ic * self.noise_sample_per_ic

        self._q_indices = self._compute_q_indices()

    def _compute_q_indices(self) -> Optional[torch.Tensor]:
        """Returns indices of humidity variables in input tensors.

        This step will later be included in the zarr building process.
        """
        q_idx = []
        for vname, vidx in self.data_indices.data.input.name_to_index.items():
            if vname.startswith("q_"):  # humidity (pl / ml)
                q_idx.append(vidx)
        LOGGER.debug("q_* indices in the input tensor: %s", 'q_idx if q_idx else "n/a"')
        return torch.IntTensor(q_idx) if q_idx else None

    def forward(self, x_an: torch.Tensor, x_ic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate initial conditions for the ensemble based on the EDA perturbations.

        
        For each IC, we sample noise_sample_per_ic times from a gaussian distribution.
        If no EDA perturbations are given, we simply stack the deterministic ERA5
        analysis nens_per_device times along the ensemble dimension.


        Inputs:
            x_an: unperturbed IC (ERA5 analysis), shape = (bs, 1, ms + rollout, latlon, v)
            x_ic : x_an with recentered ERA5 EDA perturbations, shape = (bs, ic_ens, ms, latlon, v)

        Returns
        -------
            Ensemble IC, shape (bs, ms, nens_per_device, latlon, input.full)
        """


        LOGGER.debug("EDA -- SHAPES: x_an.shape = %s, x_ic.shape = %s", list(x_an.shape), list(x_ic.shape))

        assert x_ic.shape[2] == self.multi_step and x_ic.shape[1] == self.nens_per_device
        
        if x_ic is None:
            # Sample noise_sample_per_ic times from a gaussian distribution for each IC
            outp = x_an.repeat(1, self.noise_sample_per_ic, 1, 1, 1)
        else:
            outp = x_ic.repeat(1, self.noise_sample_per_ic, 1, 1, 1)

        return outp[..., self.data_indices.data.input.full]
