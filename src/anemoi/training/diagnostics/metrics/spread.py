import logging
from typing import Optional

import torch
from torchmetrics import Metric

LOGGER = logging.getLogger(__name__)


class SpreadSkill(Metric):
    """Calculates spread skill."""

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True

    def __init__(self, rollout: int, nvar: int, nbins: int) -> None:
        """Initialise class.

        Args:
            rollout: length of rollout window
            nvar: number of dynamical variables (we produce one spread-skill diagram per variable)
            nbins: number of bins used to calculate binned rmse and spread
            node_weights: pixel area weights
        """
        super().__init__()

        self.rollout = rollout
        self.nvar = nvar
        self.nbins = nbins

        # TODO: this shouldn't be hardcoded
        self.time_step = 6  # fixed, for now (validation only)
        LOGGER.debug("Setting up a SpreadSkill metric with rollout = %d, nvar = %d, time_step = %d", rollout, nvar, self.time_step)

        self.add_state("num_updates", default=torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum")
        # RMSE of ensemble mean
        self.add_state("rmse", default=torch.zeros(rollout, nvar, dtype=torch.float32), dist_reduce_fx="sum")
        # Ensemble spread
        self.add_state("spread", default=torch.zeros(rollout, nvar, dtype=torch.float32), dist_reduce_fx="sum")
        # Bins RMSE
        self.add_state("bins_rmse", default=torch.zeros(rollout, nvar, nbins - 1, dtype=torch.float32), dist_reduce_fx="sum")
        # Bins Spread
        self.add_state("bins_spread", default=torch.zeros(rollout, nvar, nbins - 1, dtype=torch.float32), dist_reduce_fx="sum")

    def calculate_spread_skill(self, y_pred_denorm, y_denorm, pidx, node_weights) -> torch.Tensor:
        """Calculate the spread of the ensemble and the skill of the ensemble mean.

        Outputs the global average as well as average spread and skills of the ordered
        bins.

        Args:
            y_pred_denorm: Denormalised predictions (bs, nens_input, latlon, nvar)
            y_denorm: Denormalised truth (i.e. target fields) (bs, nens_target, latlon, nvar)
            pidx: index of variable being plotted
            node_weights: lat-lon weighting

        Return:
            skill, spread, bins for skill, bins for spread
        """

        rmse_spatial = torch.square(y_pred_denorm[..., pidx : pidx + 1].mean(dim=1) - y_denorm[..., pidx : pidx + 1].mean(dim=1))
        weighted_rmse = rmse_spatial[:, :, 0] * node_weights
        weighted_rmse /= torch.sum(node_weights.expand_as(weighted_rmse))

        spread_spatial = (
            torch.square(y_pred_denorm[..., pidx : pidx + 1] - y_pred_denorm[..., pidx : pidx + 1].mean(dim=1, keepdim=True))
        ).mean(dim=1)
        weighted_spread = spread_spatial[:, :, 0] * node_weights
        weighted_spread /= torch.sum(node_weights.expand_as(weighted_spread))

        spread_err = (
            torch.concat(
                [
                    weighted_spread.reshape((weighted_spread.shape[0] * weighted_spread.shape[1], 1)),
                    weighted_rmse.reshape((weighted_rmse.shape[0] * weighted_rmse.shape[1], 1)),
                ],
                axis=-1,
            )
            .sort(dim=0)
            .values
        )

        bin_width = int(len(spread_err) / (self.nbins - 1))

        # Split ordered spread-err into bins of equal width.
        # Then take sum and sqrt to find the average spread and error of each bin for plotting
        bins_spread = torch.Tensor([float(torch.sqrt(i.sum())) for i in torch.split(spread_err[:, 0], bin_width)][:-1])
        bins_rmse = torch.Tensor([float(torch.sqrt(i.sum())) for i in torch.split(spread_err[:, 1], bin_width)][:-1])

        return torch.sqrt(weighted_rmse.sum()), torch.sqrt(weighted_spread.sum()), bins_rmse, bins_spread

    def update(
        self,
        rmse: torch.Tensor,
        spread: torch.Tensor,
        bins_rmse: torch.tensor,
        bins_spread: torch.tensor,
        device: "str",
    ) -> None:
        """Update spread skill.

        Args:
            rmse: shape (rollout, nvar)
            spread: shape (rollout, nvar)
            bins_rmse: shape (rollout, nvar, nbins)
            bins_spread: shape (rollout, nvar, nbins)
            device: torch device
        """
        assert rmse.shape == (
            self.rollout,
            self.nvar,
        ), f"Shape mismatch: expected {self.rmse.shape} and got {rmse.shape}"
        assert spread.shape == (
            self.rollout,
            self.nvar,
        ), f"Shape mismatch: expected {self.spread.shape} and got {spread.shape}"

        assert bins_rmse.shape == (
            self.rollout,
            self.nvar,
            self.nbins - 1,
        ), f"Shape mismatch: expected {self.bins_rmse.shape} and got {bins_rmse.shape}"
        assert bins_spread.shape == (
            self.rollout,
            self.nvar,
            self.nbins - 1,
        ), f"Shape mismatch: expected {self.bins_spread.shape} and got {bins_spread.shape}"

        if self.rmse.device != rmse.device:
            self.rmse = self.rmse.to(device=device)
            self.spread = self.spread.to(device=device)
            self.bins_rmse = self.bins_rmse.to(device=device)
            self.bins_spread = self.bins_spread.to(device=device)
            self.num_updates = self.num_updates.to(device=device)

        self.rmse += rmse
        self.spread += spread
        self.bins_rmse += bins_rmse
        self.bins_spread += bins_spread
        self.num_updates += 1

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.rmse / self.num_updates,
            self.spread / self.num_updates,
            self.bins_rmse / self.num_updates,
            self.bins_spread / self.num_updates,
        )
