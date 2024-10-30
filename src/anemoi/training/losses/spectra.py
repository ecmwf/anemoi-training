import torch
from torch import nn
import torch_harmonics as harmonics
from typing import Optional
import einops
from .mixins import TargetEachEnsIndepMixin
from typing import Union
#TODO(rilwan-ade): Probably need to add optional weightings for frequencies since this will just focus on the larger scales
#TODO(rilwan-ade): maybe make loss across amplitudes proportional difference for scales - or (scale it proportional to how it generally varies for each variable)

class SHTBaseLoss(TargetEachEnsIndepMixin, nn.Module):
    """Base class for spectral losses using Spherical Harmonic Transforms."""

    map_spectral_loss_impl_to_group_on_dim = {
        "2D_spatial": (-2),
        "1D_temporal": (-3,),
        "3D_spatiotemporal": (-3, -2),
    }

    def __init__(
        self,
        node_weights: torch.Tensor,
        nlat: int,
        nlon: int,
        grid: str = "legendre-gauss",
        lmax: Optional[int] = None,
        feature_weights: Optional[torch.Tensor] = None,
        spectral_loss_method: str = "2D_spatial",
        target_each_ens_indep: bool = False,
        ignore_nans: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.spectral_loss_method = spectral_loss_method
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum
        self.nlat = nlat
        self.nlon = nlon
        self.solver = None
        self.target_each_ens_indep = target_each_ens_indep

        if spectral_loss_method != "1D_temporal":
            if grid not in ["legendre-gauss", "lobatto", "equiangular"]:
                msg = f"Unsupported grid type: {grid}"
                raise ValueError(msg)
            self.solver = harmonics.RealSHT(self.nlat, self.nlon, lmax=lmax, grid=grid, csphase=True)

        # Register area and feature weights
        self.register_buffer("node_weights", node_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)

        # Map the spectral loss method to its corresponding function
        self.spectral_loss_impl = {
            "2D_spatial": self._spectral_loss_2D,
            "1D_temporal": self._spectral_loss_1D_temporal,
            "3D_spatiotemporal": self._spectral_loss_3D_spatiotemporal,
        }
        self.group_on_dim: tuple = self.map_spectral_loss_impl_to_group_on_dim[spectral_loss_method]

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        squash: Union[bool, tuple] = True,
        feature_scale: bool = True,
        feature_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the area-weighted and feature-weighted spectral loss."""
        # Apply spectral loss calculation
        loss = self._spectral_loss(preds, target)

        # Average across ensemble dimension
        loss = loss.mean(dim=1)

        # Apply feature scaling if required
        if feature_scale and self.feature_weights is not None:
            if feature_indices is None:
                loss = loss * self.feature_weights
            else:
                loss = loss * self.feature_weights[..., feature_indices]
            loss = loss / self.feature_weights.numel()

        # Apply area (spatial) weighting
        loss *= (self.node_weights / self.sum_function(self.node_weights))

        # Squash (reduce spatial and feature dimensions)
        if squash:
            # Since this loss removes a dim, we have to adjust the squash_dims appropriately considering where the removed dim was
            dim = tuple( (dim if dim>max(self.group_on_dim) else dim+len(self.group_on_dim)) for dim in dim if dim not in self.group_on_dim ) if isinstance(squash, tuple) else (range(-1, -4+len(self.group_on_dim) ,-1)) 
            
            output = self.sum_function(output, dim=dim)

        return loss.mean(dim=0)

    def _spectral_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the spectral loss based on the specified spectral method."""
        return self.spectral_loss_impl[self.spectral_loss_method](preds, target)

    # Placeholder methods to be implemented by subclasses
    def _spectral_loss_2D(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _spectral_loss_1D_temporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _spectral_loss_3D_spatiotemporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpectralEnergyLoss(SHTBaseLoss):
    """Spectral Energy Loss with area- and feature-weighting support."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        nlat: int,
        nlon: int,
        grid: str = "legendre-gauss",
        lmax: Optional[int] = None,
        feature_weights: Optional[torch.Tensor] = None,
        beta: int = 2,
        spectral_loss_method: str = "2D_spatial",
        target_each_ens_indep: bool = False,
        ignore_nans: Optional[bool] = False,
    ) -> None:
        super().__init__(
            node_weights,
            nlat,
            nlon,
            grid,
            lmax,
            feature_weights,
            spectral_loss_method,
            target_each_ens_indep,
            ignore_nans,
        )
        self.beta = beta

    def _spectral_loss_2D(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape tensors
        preds = einops.rearrange(
            preds,
            "bs ens_inp timesteps (lat lon) nvars -> bs ens_inp timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )
        target = einops.rearrange(
            target,
            "bs ens_target timesteps (lat lon) nvars -> bs ens_target timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )

        # Apply SHT
        preds_sht = self.solver(preds)
        target_sht = self.solver(target)

        # Compute energies (magnitude)
        preds_energy = torch.abs(preds_sht)
        target_energy = torch.abs(target_sht)

        # Average across ensemble dimension
        preds_energy = preds_energy.mean(dim=1)
        target_energy = target_energy.mean(dim=1)

        # Compute energy difference
        energy_diff = preds_energy - target_energy

        # Apply beta power
        power = energy_diff.abs() ** self.beta

        # Rearrange back
        output = einops.rearrange(
            power,
            "bs timesteps nvars lat lon -> bs timesteps (lat lon) nvars",
        )

        return output

    def _spectral_loss_1D_temporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply 1D temporal FFT
        preds_fft = torch.fft.fftn(preds, dim=-3)
        target_fft = torch.fft.fftn(target, dim=-3)

        # Compute energies (magnitude)
        preds_energy = torch.abs(preds_fft)
        target_energy = torch.abs(target_fft)

        # Average across ensemble dimension
        preds_energy = preds_energy.mean(dim=1)
        target_energy = target_energy.mean(dim=1)

        # Compute energy difference
        energy_diff = preds_energy - target_energy

        # Apply beta power
        output = energy_diff.abs() ** self.beta

        return output

    def _spectral_loss_3D_spatiotemporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape tensors
        preds = einops.rearrange(
            preds,
            "bs ens_inp timesteps (lat lon) nvars -> bs ens_inp timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )
        target = einops.rearrange(
            target,
            "bs ens_target timesteps (lat lon) nvars -> bs ens_target timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )

        # Apply SHT
        preds_sht = self.solver(preds)
        target_sht = self.solver(target)

        # Apply FFT for temporal dimension
        preds_sht_fft = torch.fft.fftn(preds_sht, dim=-3)
        target_sht_fft = torch.fft.fftn(target_sht, dim=-3)

        # Compute energies (magnitude)
        preds_energy = torch.abs(preds_sht_fft)
        target_energy = torch.abs(target_sht_fft)

        # Average across ensemble dimension
        preds_energy = preds_energy.mean(dim=1)
        target_energy = target_energy.mean(dim=1)

        # Compute energy difference
        energy_diff = preds_energy - target_energy

        # Apply beta power
        output = energy_diff.abs() ** self.beta

        return output

class SHTAmplitudePhaseLoss(SHTBaseLoss):
    """Spectral loss that combines amplitude and phase differences."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        nlat: int,
        nlon: int,
        grid: str = "legendre-gauss",
        lmax: Optional[int] = None,
        feature_weights: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        spectral_loss_method: str = "2D_spatial",
        target_each_ens_indep: bool = False,
        ignore_nans: Optional[bool] = False,
    ) -> None:
        super().__init__(
            node_weights,
            nlat,
            nlon,
            grid,
            lmax,
            feature_weights,
            spectral_loss_method,
            target_each_ens_indep,
            ignore_nans,
        )
        self.alpha = alpha  # Weight for amplitude loss
        self.beta = beta    # Weight for phase loss

    def _compute_amplitude_phase_loss(self, preds_complex, target_complex):
        # Compute amplitude (magnitude) and phase (angle)
        preds_amplitude = torch.abs(preds_complex)
        target_amplitude = torch.abs(target_complex)

        preds_phase = torch.angle(preds_complex)
        target_phase = torch.angle(target_complex)

        # Compute amplitude difference
        amplitude_diff = preds_amplitude - target_amplitude
        amplitude_loss = amplitude_diff ** 2

        # Compute phase difference, mapping to [-pi, pi]
        phase_diff = preds_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        phase_loss = phase_diff ** 2

        # Combine losses
        total_loss = self.alpha * amplitude_loss + self.beta * phase_loss

        return total_loss

    def _spectral_loss_2D(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape tensors
        preds = einops.rearrange(
            preds,
            "bs ens_inp timesteps (lat lon) nvars -> bs ens_inp timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )
        target = einops.rearrange(
            target,
            "bs ens_target timesteps (lat lon) nvars -> bs ens_target timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )

        # Apply SHT
        preds_sht = self.solver(preds)
        target_sht = self.solver(target)

        # Compute loss
        loss = self._compute_amplitude_phase_loss(preds_sht, target_sht)

        # Average across ensemble dimension
        loss = loss.mean(dim=1)

        # Rearrange back
        output = einops.rearrange(
            loss,
            "bs timesteps nvars lat lon -> bs timesteps (lat lon) nvars",
        )

        return output

    def _spectral_loss_1D_temporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply FFT for temporal dimension
        preds_fft = torch.fft.fftn(preds, dim=-3)
        target_fft = torch.fft.fftn(target, dim=-3)

        # Compute loss
        loss = self._compute_amplitude_phase_loss(preds_fft, target_fft)

        # Average across ensemble dimension
        loss = loss.mean(dim=1)

        return loss

    def _spectral_loss_3D_spatiotemporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape tensors
        preds = einops.rearrange(
            preds,
            "bs ens_inp timesteps (lat lon) nvars -> bs ens_inp timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )
        target = einops.rearrange(
            target,
            "bs ens_target timesteps (lat lon) nvars -> bs ens_target timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )

        # Apply SHT
        preds_sht = self.solver(preds)
        target_sht = self.solver(target)

        # Apply FFT for temporal dimension
        preds_sht_fft = torch.fft.fftn(preds_sht, dim=-3)
        target_sht_fft = torch.fft.fftn(target_sht, dim=-3)

        # Compute loss
        loss = self._compute_amplitude_phase_loss(preds_sht_fft, target_sht_fft)

        # Average across ensemble dimension
        loss = loss.mean(dim=1)

        return loss

class SHTComplexBetaLoss(SHTBaseLoss):
    """Spectral loss computed directly on complex coefficients with customizable power."""
    
    def __init__(
        self,
        node_weights: torch.Tensor,
        nlat: int,
        nlon: int,
        grid: str = "legendre-gauss",
        lmax: Optional[int] = None,
        feature_weights: Optional[torch.Tensor] = None,
        beta: float = 2.0,
        spectral_loss_method: str = "2D_spatial",
        target_each_ens_indep: bool = False,
        ignore_nans: Optional[bool] = False,
    ) -> None:
        super().__init__(
            node_weights,
            nlat,
            nlon,
            grid,
            lmax,
            feature_weights,
            spectral_loss_method,
            target_each_ens_indep,
            ignore_nans,
        )
        self.beta = beta  # Power to raise the absolute difference

    def _compute_complex_power_loss(self, preds_complex, target_complex):
        # Compute complex difference
        complex_diff = preds_complex - target_complex
        # Compute the loss using the specified beta power
        complex_loss = torch.abs(complex_diff) ** self.beta
        return complex_loss

    def _spectral_loss_2D(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape tensors
        preds = einops.rearrange(
            preds,
            "bs ens_inp timesteps (lat lon) nvars -> bs ens_inp timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )
        target = einops.rearrange(
            target,
            "bs ens_target timesteps (lat lon) nvars -> bs ens_target timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )

        # Apply SHT
        preds_sht = self.solver(preds)
        target_sht = self.solver(target)

        # Compute loss
        loss = self._compute_complex_power_loss(preds_sht, target_sht)

        # Average across ensemble dimension
        loss = loss.mean(dim=1)

        # Rearrange back
        output = einops.rearrange(
            loss,
            "bs timesteps nvars lat lon -> bs timesteps (lat lon) nvars",
        )

        return output

    def _spectral_loss_1D_temporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply FFT for temporal dimension
        preds_fft = torch.fft.fftn(preds, dim=-3)
        target_fft = torch.fft.fftn(target, dim=-3)

        # Compute loss
        loss = self._compute_complex_power_loss(preds_fft, target_fft)

        # Average across ensemble dimension
        loss = loss.mean(dim=1)

        return loss

    def _spectral_loss_3D_spatiotemporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape tensors
        preds = einops.rearrange(
            preds,
            "bs ens_inp timesteps (lat lon) nvars -> bs ens_inp timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )
        target = einops.rearrange(
            target,
            "bs ens_target timesteps (lat lon) nvars -> bs ens_target timesteps nvars lat lon",
            lat=self.nlat,
            lon=self.nlon,
        )

        # Apply SHT
        preds_sht = self.solver(preds)
        target_sht = self.solver(target)

        # Apply FFT for temporal dimension
        preds_sht_fft = torch.fft.fftn(preds_sht, dim=-3)
        target_sht_fft = torch.fft.fftn(target_sht, dim=-3)

        # Compute loss
        loss = self._compute_complex_power_loss(preds_sht_fft, target_sht_fft)

        # Average across ensemble dimension
        loss = loss.mean(dim=1)

        return loss
