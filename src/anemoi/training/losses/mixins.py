from typing import Optional

import einops
import torch
from typing import Union

class TargetEachEnsIndepMixin:
    """Mixin to handle target_each_ens_indep logic for SpectralEnergyLoss.
        #TODO(rilwan-ade): This mixin should apply to every single deterministic loss 
            should be refactored to be a flag between learn_ensemble_mean vs learn_each_ens_indep
    """

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: Union[bool, tuple] = True,
        feature_scale: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Wraps the forward method to handle target_each_ens_indep logic.

        If target_each_ens_indep is True, it folds the ensemble dimension into the batch dimension
        and adjusts the scaling of the output loss accordingly.
        """
        if self.target_each_ens_indep:
            # Fold the ensemble dimension into the batch dimension
            assert preds.shape[1] == target.shape[1], "Ensemble sizes must be equal"
            
            ens_inp = preds.shape[1]
            
            preds = einops.rearrange(
                preds, "bs ens timesteps latlon nvars -> (bs ens) 1 timesteps latlon nvars"
            )
            target = einops.rearrange(
                target, "bs ens timesteps latlon nvars -> (bs ens) 1 timesteps latlon nvars"
            )
            # Call the original forward method
            loss = super().forward(preds, target, squash, feature_scale, feature_indices)
            # Scale up the loss to account for the folded ensemble dimension
            loss = loss * ens_inp  # Assuming self.n_ens holds the ensemble size
            return loss
        else:
            # Call the original forward method without modification
            return super().forward(preds, target, squash, feature_scale, feature_indices)