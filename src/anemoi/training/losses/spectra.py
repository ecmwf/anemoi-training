import torch
from torch import nn
import torch_harmonics as harmonics
from typing import Optional

class SpectralEnergyLoss(nn.Module):
    """Spectral Energy Loss with area- and feature-weighting support."""

    def __init__(
        self,
        area_weights: torch.Tensor,
        nlat: int,
        nlon: int,
        grid: str = 'legendre-gauss',
        lmax: Optional[int] = None,
        feature_weights: Optional[torch.Tensor] = None,
        power: int = 2,
        spectral_loss_method: str = "2D_spatial",
        frequency_weighting_method: str = "none",
        ignore_nans: Optional[bool] = False,
    ) -> None:
        """Initialize the Spectral Energy Loss module with a configurable power term.

        Args:
            area_weights : torch.Tensor
                Weights by area (latitude or spatial domain).
            nlat : int
                Number of latitudinal points.
            nlon : int
                Number of longitudinal points.
            grid : str, optional
                Type of grid ('legendre-gauss' or 'icosahedral'), by default 'legendre-gauss'.
            lmax : Optional[int], optional
                Maximum degree of spherical harmonics, by default None.
            feature_weights : Optional[torch.Tensor], optional
                Loss weighting by feature (e.g., different variables), by default None.
            power : int, optional
                Power applied to the spectral energy, by default 2.
            spectral_loss_method : str, optional
                Method for calculating spectral loss, by default "2D_spatial".
            frequency_weighting_method : str, optional
                Method to apply frequency weighting to the spectral loss, by default "none".
            ignore_nans : bool, optional
                Allow NaNs in the loss calculation, by default False.
        """
        super().__init__()
        self.power = power
        self.spectral_loss_method = spectral_loss_method
        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum
        self.solver = None

        if spectral_loss_method != "1D_temporal":
            if grid not in ["legendre-gauss", "lobatto", "equiangular"]:
                # For any of the spatial grids, must use the FXXX grids which are legender-gauss
                raise ValueError(f"Unsupported grid type: {grid}")
                    # Create solver based on grid type and initialize spherical harmonics transformer
            self.solver = harmonics.RealSHT(nlat, nlon, lmax=lmax, grid=grid, csphase=False)
            
        

        # Register area and feature weights
        self.register_buffer("area_weights", area_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)



        # Map the spectral loss method to its corresponding function
        self.spectral_loss_impl = {
            "2D_spatial": self._spectral_energy_2D,
            "1D_temporal": self._spectral_energy_1D_temporal,
            "3D_spatiotemporal": self._spectral_energy_3D_spatiotemporal
        }

        #TODO (rilwan-ade): Implement frequency weighting

    def _spectral_energy(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the spectral energy loss based on the spectral method."""
        return self.spectral_loss_impl[self.spectral_loss_method](preds, target)

    def _spectral_energy_2D(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 2D spatial spherical harmonic spectral energy."""
        # Spherical Harmonic Transform (SHT) for spatial dimensions
        preds_sht = self.solver.grid2spec(preds)
        target_sht = self.solver.grid2spec(target)

        preds_energy = torch.abs(preds_sht)
        target_energy = torch.abs(target_sht)
        energy_diff = preds_energy - target_energy

        return energy_diff.abs() ** self.power

    def _spectral_energy_1D_temporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1D temporal FFT-based spectral energy."""
        # 1D temporal FFT
        preds_fft = torch.fft.fftn(preds, dim=-3)
        target_fft = torch.fft.fftn(target, dim=-3)

        preds_energy = torch.abs(preds_fft)
        target_energy = torch.abs(target_fft)
        energy_diff = preds_energy - target_energy

        return energy_diff.abs() ** self.power

    def _spectral_energy_3D_spatiotemporal(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D spatiotemporal spectral energy using spherical harmonics for spatial dimensions and FFT for the temporal dimension."""
        # Spherical Harmonic Transform (SHT) for spatial dimensions
        preds_sht = self.solver.grid2spec(preds)
        target_sht = self.solver.grid2spec(target)

        # FFT for temporal dimension
        preds_fft = torch.fft.fftn(preds_sht, dim=-3)
        target_fft = torch.fft.fftn(target_sht, dim=-3)

        preds_energy = torch.abs(preds_fft)
        target_energy = torch.abs(target_fft)
        energy_diff = preds_energy - target_energy

        return energy_diff.abs() ** self.power

    def forward(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            squash: bool = True,
            feature_scale: bool = True,
            feature_indices: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Calculate the area-weighted and feature-weighted spectral energy loss.

            Args:
                preds : torch.Tensor shape (bs, ens, (timesteps), lat*lon, n_outputs)
                    Predicted values.
                target : torch.Tensor shape (bs, (timesteps), lat*lon, n_outputs)
                    Ground truth values.
                squash : bool, optional
                    Whether to reduce spatial and feature dimensions, by default True.
                feature_scale : bool, optional
                    Whether to apply feature scaling, by default True.
                feature_indices: Optional[torch.Tensor], optional
                    Indices to scale the loss by specific features, by default None.

            Returns:
                torch.Tensor
                    The computed weighted spectral energy loss.
            """
            # Apply spectral energy calculation
            loss = self._spectral_energy(preds, target)

            # Apply feature scaling if required
            if feature_scale and self.feature_weights is not None:
                loss = loss * self.feature_weights if feature_indices is None else loss * self.feature_weights[..., feature_indices]
                loss = loss / self.feature_weights.numel()

            # Apply area (spatial) weighting
            loss *= (self.area_weights / self.sum_function(self.area_weights))

            # Squash (reduce spatial and feature dimensions)
            if squash:
                loss = loss.sum(dim=(-2, -1))

            return loss.mean(dim=0)
