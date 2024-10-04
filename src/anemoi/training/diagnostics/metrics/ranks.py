import logging
from typing import Optional

import torch
from torchmetrics import Metric

LOGGER = logging.getLogger(__name__)


def get_ranks(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Gets ranks for histogram.

    Args:
        truth: shape (bs, latlon, nvar)
        pred: shape (bs, nens, latlon, nvar)

    Return:
        ranks: mask array where truth > pred, count
    """
    return torch.count_nonzero(truth[:, None, ...] >= pred, dim=1)


# class RankHistogram(Metric):
#     """Build a rank histogram."""

#     is_differentiable: bool = False
#     higher_is_better: Optional[bool] = None
#     full_state_update: bool = True

#     def __init__(self, nens: int, nvar: int) -> None:
#         """Initialise class.

#         Args:
#             nens: size of predicted ensemble; we'll have n_ens + 1 bins in our rank histogram
#             nvar: number of physical variables; each will get its own rank histogram
#         """
#         super().__init__()

#         self.nens = nens
#         self.nvar = nvar
#         self.add_state("ranks", default=torch.zeros(nens + 1, nvar, dtype=torch.long), dist_reduce_fx="sum")

#     def update(self, truth: torch.Tensor, pred: torch.Tensor, device: "str") -> None:
#         """Update ranks.

#         Args:
#             truth: shape (bs, latlon, nvar)
#             pred: shape (bs, nens, latlon, nvar)
#             device: torch device
#         """
#         ranks_ = get_ranks(truth, pred).flatten(end_dim=1)
#         # update the running stats
#         # NB: this will calculate a running sum instead of the accumulated totals
#         if self.ranks.device != ranks_.device:
#             self.ranks = self.ranks.to(device=device)

#         for ivar in range(self.nvar):
#             self.ranks[:, ivar] += ranks_[:, ivar].flatten().bincount(minlength=self.nens + 1)

#     def compute(self):
#         return self.ranks.float() / self.ranks.sum(dim=0, keepdim=True)

# NOTE: Updated to work with sampled observations
class RankHistogram(Metric):
    """Build a continuous rank histogram."""

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True

    def __init__(self, nens_input: int, nvar: int) -> None:
        """Initialise class.

        Args:
            nens_input: size of predicted ensemble; we'll have n_ens + 1 bins in our rank histogram
            nvar: number of physical variables; each will get its own rank histogram
        """
        super().__init__()

        self.nens_input = nens_input
        self.nvar = nvar
        self.add_state("ranks", default=torch.zeros(nens_input + 1, nvar, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, truth: torch.Tensor, pred: torch.Tensor, device: "str") -> None:
        """Update ranks with continuous observation values.

        Args:
            truth: shape (bs, nens_target, latlon, nvar) - truth distribution or values
            pred: shape (bs, nens_input, latlon, nvar) - ensemble forecast distribution (or single value if nens=1)
            device: torch device
        """
        # Check if the truth is a single value or a sample
        if truth.shape[1] != 1:  # Single value case
            ranks_ = self.get_single_value_ranks(truth, pred).flatten(end_dim=1)
        else:  # Sample case
            ranks_ = self.get_sample_ranks(truth, pred).flatten(end_dim=1)
        
        # Update the running stats
        if self.ranks.device != ranks_.device:
            self.ranks = self.ranks.to(device=device)

        for ivar in range(self.nvar):
            self.ranks[:, ivar] += ranks_[:, ivar].flatten().bincount(minlength=self.nens_input + 1)

    def get_single_value_ranks(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Calculate rank when the truth is a single value.

        Args:
            truth: shape (bs, latlon, nvar)
            pred: shape (bs, nens_input, latlon, nvar)

        Returns:
            Ranks: a tensor of ranks (0 if truth < pred, 1 if truth >= pred)
        """
        truth = truth.squeeze(1)
        ranks = torch.zeros_like(truth, dtype=torch.long)
        ranks[truth >= pred.squeeze(1)] = 1  # Rank 1 if truth >= prediction, else rank 0
        return ranks

    def get_sample_ranks(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Calculate the rank when the truth is a sample.

        Args:
            truth: shape (bs, nens_target, latlon, nvar)
            pred: shape (bs, nens_input, latlon, nvar)

        Returns:
            Ranks: a tensor of ranks based on CDF evaluation
        """
        cdf_values = torch.zeros_like(truth[:, 0, ...])
        for i in range(self.nens_input):  # iterate over ensemble members
            cdf_values += (pred[:, i, ...] <= truth).float().mean(dim=1)

        # Normalize the rank based on the number of ensemble members
        cdf_values /= pred.size(1)
        return cdf_values

# if __name__ == "__main__":
#     bs, v, nlatlon, e = 4, 28, 256, 8
#     node_weights = torch.ones(nlatlon, dtype=torch.float32)
#     metric = RankHistogram(e, v)

#     n_batches = 10
#     for _ in range(n_batches):
#         yt = torch.randn(bs, nlatlon, v)
#         yp = torch.randn(bs, e, nlatlon, v)  # perfectly calibrated (uniform)
#         # yp = 2 * torch.randn(bs, e, nlatlon, v)  # overdispersive - "peaked"
#         # yp = 0.25 * torch.randn(bs, e, nlatlon, v)  # underdispersive - u-shaped
#         # yp = 0.5 * torch.abs(torch.randn(bs, e, nlatlon, v))  # strong skew to the left, i.e. underforecasting
#         # yp = -0.5 * torch.abs(torch.randn(bs, e, nlatlon, v))  # strong skew to the right, i.e. overforecasting
#         rh = metric(yt, yp)

#     rh = metric.compute()
#     assert rh.shape == (e + 1, v)
#     torch.set_printoptions(precision=3)
#     for iv in range(v):
#         LOGGER.debug("Rank histogram: %s -- sum: %.2e", rh[:, iv], rh[:, iv].sum())
