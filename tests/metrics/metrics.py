import torch
import pytest
from src.anemoi.training.diagnostics.metrics.ranks import RankHistogram

@pytest.mark.parametrize("bs, nlatlon, nvar, nens_input", [(4, 256, 28, 4)])
def test_rank_histogram_single_value(bs, nlatlon, nvar, nens_input):
    """Test RankHistogram with a single prediction (nens = 1)."""
    metric = RankHistogram(nens_input, nvar)

    # Create deterministic data for testing
    truth = torch.linspace(0, 1, bs * nlatlon * nvar).reshape(bs, 1, nlatlon, nvar)
    pred = torch.full((bs, nens_input, nlatlon, nvar), 0.5)

    metric.update(truth, pred, device='cpu')
    rh = metric.compute()

    assert rh.shape == (nens_input + 1, nvar), "Rank histogram shape mismatch for single prediction case."
    
    # Check correctness: half of the values should be in each bin
    expected_counts = torch.tensor([bs * nlatlon * nvar // 2, bs * nlatlon * nvar // 2]).float()
    expected_histogram = expected_counts / expected_counts.sum()
    
    assert torch.allclose(rh[:, 0], expected_histogram), "Incorrect rank histogram for single prediction case."

@pytest.mark.parametrize("bs, nlatlon, nvar, nens_input, nens_target", [(4, 256, 28, 8, 5)])
def test_rank_histogram_sample(bs, nlatlon, nvar, nens_input, nens_target):
    """Test RankHistogram with sampled truth values (nens_target > 1)."""
    metric = RankHistogram(nens_input, nvar)

    # Create deterministic data for testing
    truth = torch.linspace(0, 1, bs * nens_target * nlatlon * nvar).reshape(bs, nens_target, nlatlon, nvar)
    pred = torch.linspace(0, 1, nens_input)[None, :, None, None].repeat(bs, 1, nlatlon, nvar)

    metric.update(truth, pred, device='cpu')
    rh = metric.compute()

    assert rh.shape == (nens_input + 1, nvar), "Rank histogram shape mismatch for sampled truth case."
    
    # Check correctness: expect uniform distribution across bins
    expected_histogram = torch.full((nens_input + 1,), 1 / (nens_input + 1))
    
    assert torch.allclose(rh[:, 0], expected_histogram, atol=1e-6), "Incorrect rank histogram for sampled truth case."