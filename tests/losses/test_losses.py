import einops
import numpy as np
import pytest
import scoringrules as sr
import torch
from omegaconf import OmegaConf
from anemoi.training.losses import (
    EnergyScore, GroupedEnergyScore, KLDivergenceLoss, RenyiDivergenceLoss,
    WeightedMSELoss, WeightedMAELoss, VAELoss, VariogramScore, SpectralEnergyLoss, kCRPS, IgnoranceScore,
    CompositeLoss, SpreadLoss, SpreadSkillLoss, ZeroSpreadRateLoss
)
import torch_harmonics as th
from scipy.stats import norm  # Add this import
from numpy.fft import fftn  # Add this import
import numpy as np
from anemoi.training.losses import MultivariatekCRPS, GroupedMultivariatekCRPS
import logging
LOGGER = logging.getLogger(__name__)
import copy
import lovely_tensors as lt
lt.monkey_patch()
import typing as tp
from lovely_numpy import lo
if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array
from omegaconf import DictConfig

import hydra

class TestLosses:
    @classmethod
    def setup_class(cls) -> None:
        cls.eps = 1e-5  # Set epsilon to 0 for testing
        LOGGER.info("Setup TestLosses class")

    @classmethod
    def teardown_class(cls) -> None:
        """Runs once after all tests in the class."""
        LOGGER.info("Teardown TestLosses class")

    def setup_method(self, method) -> None:
        """Runs before every test method in the class."""
        LOGGER.info(f"Setup {method.__name__}")

    def teardown_method(self, method) -> None:
        """Runs after every test method in the class."""
        LOGGER.info(f"Teardown {method.__name__}")

    @pytest.mark.parametrize(
        "group_on_dim, nens_input, nens_output",
        [
            (-1, 4, 1 ),
            (-2, 4, 1),
            (-3, 4, 1),
            (-1, 4, 2 ),
            (-2, 4, 2),
            (-3, 4, 2),
        ],
    )
    def test_energy_score_forward(self, group_on_dim: int, nens_input: int, nens_output: int) -> None:
        bs, timesteps, latlon, nvars = 2, 6, 40320, 80
        
        preds = torch.randn(bs, nens_input, timesteps, latlon, nvars, dtype=torch.float32)
        target = torch.randn(bs, nens_output, timesteps, latlon, nvars, dtype=torch.float32)

        node_weights = 0.5 + torch.rand(latlon, dtype=torch.float32)
        feature_weights = 0.5 + torch.rand(nvars, dtype=torch.float32)
        beta = torch.as_tensor(1.0)
        
        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()
        beta_np = beta.numpy()


        energy_score = EnergyScore(
            node_weights=node_weights,
            feature_weights=feature_weights,
            group_on_dim=group_on_dim,
            beta=beta,
            fair=False,
            )

        score = energy_score.forward(
            preds,
            target,
            squash=False,
        )
        score = score.numpy()

        # Reference calculation adjustments for EnergyScore specifics


        # Use NumPy arrays for reference calculation
        if group_on_dim == -2:
            forecasts = preds_np * ((node_weights_np[..., None] / node_weights_np.sum()) ** (1 / beta_np))
            observations = target_np * ((node_weights_np[..., None] / node_weights_np.sum()) ** (1 / beta_np))

            forecasts, observations = sr.core.utils.multivariate_array_check(
                forecasts,
                observations,
                1,
                group_on_dim,
                backend="numpy",
            )

            reference_score = sr.core.energy.nrg(forecasts, observations, backend="numpy")
            reference_score = reference_score * feature_weights_np/feature_weights_np.size

        elif group_on_dim == -1:
            forecasts = preds_np * ((feature_weights_np/feature_weights_np.size) ** (1 / beta_np))
            observations = target_np * ((feature_weights_np/feature_weights_np.size) ** (1 / beta_np))

            forecasts, observations = sr.core.utils.multivariate_array_check(
                forecasts,
                observations,
                1,
                group_on_dim,
                backend="numpy",
            )

            reference_score = sr.core.energy.nrg(forecasts, observations, backend="numpy")
            reference_score = reference_score * (node_weights_np / node_weights_np.sum())
        
        elif group_on_dim == -3:
            forecasts = preds_np * ((feature_weights_np/feature_weights_np.size) ** (1 / beta_np))
            observations = target_np * ((feature_weights_np/feature_weights_np.size) ** (1 / beta_np))

            forecasts, observations = sr.core.utils.multivariate_array_check(
                forecasts,
                observations,
                1,
                group_on_dim,
                backend="numpy",
            )

            reference_score = sr.core.energy.nrg(forecasts, observations, backend="numpy")
            reference_score = reference_score * (node_weights_np[..., None] / node_weights_np.sum())

        reference_score = np.mean(reference_score, 0)
        np.testing.assert_allclose(score, reference_score, rtol=1e-2)

    @pytest.mark.parametrize(
        "patch_method, param_groups, data_indices_model_output",
        [
            (
                "group_by_variable",
                {"q": [0, 2, 4], "t": [1, 3, 5], "v": [6, 8, 11], "z": [7], "u": [9, 10]},
                {
                    "name_to_index": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "z43": 7,
                        "15v": 8,
                        "10u": 9,
                        "u_500": 10,
                        "v_900": 11,
                    },
                    "full_idxs": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "z43": 7,
                        "15v": 8,
                        "10u": 9,
                        "u_500": 10,
                        "v_900": 11,
                    },
                },
            ),
            (
                "group_by_pressurelevel",
                {"sfc": [0, 1, 6, 7], "500": [2, 3], "850": [4, 5]},
                {
                    "name_to_index": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "10u": 7,
                    },
                    "full_idxs": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "10u": 7,
                    },
                },
            ),
        ],
    )
    def test_get_patch_sets(self, patch_method, param_groups, data_indices_model_output) -> None:
        node_weights = torch.rand(10)
        feature_weights = torch.rand(10)
        data_indices_model_output = OmegaConf.create(data_indices_model_output)

        gmkcrps = GroupedMultivariatekCRPS(
            node_weights=node_weights,
            feature_weights=feature_weights,
            patch_method=patch_method,
            group_on_dim=-1,
            beta=1.0,
            p_norm=1.5,
            fair=True,
            data_indices_model_output=data_indices_model_output,
            implementation="vectorized"
        )

        patch_sets = gmkcrps._get_patch_sets(patch_method, data_indices_model_output)

        expected_patch_sets = [{key: torch.tensor(val) for key, val in param_groups.items()}]

        assert len(patch_sets) == len(expected_patch_sets)
        for patch_set, expected_patch_set in zip(patch_sets, expected_patch_sets):
            assert patch_set.keys() == expected_patch_set.keys()
            for key in patch_set.keys():
                assert torch.equal(patch_set[key], expected_patch_set[key])

        LOGGER.info(f"Patch sets for {patch_method} Test Passed")
    
    @pytest.mark.parametrize(
        "patch_method, param_groups, data_indices_model_output",
        [
            (
                "group_by_variable",
                {"q": [0, 2, 4], "t": [1, 3, 5], "v": [6, 8, 11], "z": [7], "u": [9, 10]},
                {
                    "name_to_index": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "z43": 7,
                        "15v": 8,
                        "10u": 9,
                        "u_500": 10,
                        "v_900": 11,
                    },
                    "full_idxs": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "z43": 7,
                        "15v": 8,
                        "10u": 9,
                        "u_500": 10,
                        "v_900": 11,
                    },
                },
            ),
            (
                "group_by_pressurelevel",
                {"sfc": [0, 1, 6, 7], "500": [2, 3], "850": [4, 5]},
                {
                    "name_to_index": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "10u": 7,
                    },
                    "full_idxs": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "10u": 7,
                    },
                },
            ),
        ],
    )
    def test_grouped_energy_score(self, patch_method, param_groups, data_indices_model_output) -> None:
        bs, nens, timesteps, latlon, nvars = 2, 3, 2, 4, len(data_indices_model_output["full_idxs"])

        preds = torch.randn(bs, nens, timesteps, latlon, nvars, dtype=torch.float32)
        target = torch.randn(bs, timesteps, latlon, nvars, dtype=torch.float32)
        node_weights = 0.5 + torch.rand(latlon, dtype=torch.float32)
        feature_weights = 0.5 + torch.rand(nvars, dtype=torch.float32)
        beta = 1.0

        assert beta == 1.0  # reference score uses l2Norm with beta = 1

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()
        beta_np = np.asarray(beta)

        data_indices_model_output = OmegaConf.create(data_indices_model_output)

        pnrg_cls = GroupedEnergyScore(
            node_weights=node_weights,
            feature_weights=feature_weights,
            group_on_dim=-1,
            beta=beta,
            patch_method=patch_method,
            data_indices_model_output=data_indices_model_output,
            fair=False,
        )

        score = pnrg_cls.forward(
            preds,
            target,
            squash=False,
            deterministic=True,
        )

        # Use NumPy arrays for reference calculation
        # reference_score = np.zeros((bs, latlon))
        reference_score = target.new_zeros(target.shape[:-1])

        preds_np = preds_np * ((feature_weights_np/feature_weights_np.size) ** (1 / (beta_np) ) )
        target_np = target_np * ((feature_weights_np/feature_weights_np.size) ** (1 / (beta_np) ))

        for indices in param_groups.values():
            
            group_preds = copy.deepcopy(preds_np[..., indices]) 
            group_target = copy.deepcopy(target_np[..., indices])

            forecasts, observations = sr.core.utils.multivariate_array_check(group_preds, group_target, 1, -1, backend="numpy")
            group_reference_score = sr.core.energy.nrg(forecasts, observations, backend="numpy" )
            reference_score = reference_score + group_reference_score

        reference_score = reference_score * (node_weights_np / node_weights_np.sum())
        reference_score = reference_score.mean(0)

        np.testing.assert_allclose(score, reference_score, rtol=1e-2)
        LOGGER.info(f"Grouped Energy Score {patch_method} Test Passed")

    @pytest.mark.parametrize(
        "group_on_dim, nens_input, nens_target, target_each_ens_indep",
        [
            (-1, 3, 1, False),
            (-2, 3, 3, False),
            (-3, 1, 1, False),
            (-4, 4, 4, False),

            (-1, 3, 3, True),
            (-2, 3, 3, True),
            (-3, 3, 3, True),
        ],
    )
    def test_variogram(self, group_on_dim: int, nens_input: int, nens_target: int, target_each_ens_indep: bool) -> None:
        
        bs, timesteps, latlon, nvars = 8, 2, 20, 10

        preds = torch.randn(bs, nens_input, timesteps, latlon, nvars, dtype=torch.float32)
        target = torch.randn(bs, nens_target, timesteps, latlon, nvars, dtype=torch.float32)
        node_weights = 0.5 + torch.rand(latlon, dtype=torch.float32)
        feature_weights = 0.5 + torch.rand(nvars, dtype=torch.float32)

        node_weights = torch.ones_like(node_weights)
        feature_weights = torch.ones_like(feature_weights)
    

        beta = torch.as_tensor(1.0, dtype=torch.float32)

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()
        beta_np = beta.numpy()
        
        
        preds_, target_, node_weights_, feature_weight_, beta_ = (
            preds.clone(),
            target.clone(),
            node_weights.clone(),
            feature_weights.clone(),
            beta.clone(),
        )

        variogram_score = VariogramScore(
            node_weights=node_weights_,
            feature_weights=feature_weight_,
            group_on_dim=group_on_dim,
            beta=beta_,
            ignore_nans=True,
            target_each_ens_indep=target_each_ens_indep,
            
        )

        score = variogram_score.forward(
            preds_,
            target_,
            squash=False,
            feature_scale=True,
            feature_indices=None,
        )
        score = score.numpy()

        # Use NumPy arrays for reference calculation


        if group_on_dim == -1:
            forecasts = preds_np * ( (feature_weights_np/feature_weights_np.size) ** (1 / (beta_np * 2)) )
            observations = target_np * ( (feature_weights_np/feature_weights_np.size) ** (1 / (beta_np * 2)) )
        elif group_on_dim == -2:
            forecasts = preds_np * ( ( node_weights_np[..., None]/node_weights_np.sum()) ** (1 / (beta_np * 2)) )
            observations = target_np * ( (node_weights_np[..., None]/node_weights_np.sum()) ** (1 / (beta_np * 2)) )
        else:
            forecasts = preds_np
            observations = target_np

        if nens_target == 1 and nens_input > 1:
            forecasts, observations = sr.core.utils.multivariate_array_check(forecasts, observations.mean(1), 1, group_on_dim, backend="numpy")
            reference_score = sr.core.variogram._score.variogram_score(forecasts, observations, backend="numpy", p=beta_np.item())

        elif group_on_dim == -4:
            forecasts_ = forecasts[:, None, ...]
            observations_ = observations[:, None, ...]
            forecasts_, observations_ = sr.core.utils.multivariate_array_check(forecasts_, observations_.mean(1), 1, group_on_dim, backend="numpy")

            reference_score = sr.core.variogram._score.variogram_score(forecasts_, observations_, backend="numpy", p=beta_np.item())
            
        elif not target_each_ens_indep:
            
            def ref_variogram_score_not_target_each_ens_indep(
                # Adapted from scoringrules.core.variogram._score.variogram_score for multivariate target
                fcst: "Array",  # (... M D)
                obs: "Array",  # (... M D)
                p: float = 1,
                backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
            ) -> "Array":
                """Compute the Variogram Score for a multivariate finite ensemble."""
                from scoringrules.backend import backends

                B = backends.active if backend is None else backends[backend]
                M: int = fcst.shape[-2]
                fcst_diff = B.expand_dims(fcst, -2) - B.expand_dims(fcst, -1)  # (... M D D)
                vfcts = B.sum(B.abs(fcst_diff) ** p, axis=-3) / M  # (... D D)
                
                obs_diff = B.expand_dims(obs, -2) - B.expand_dims(obs, -1)  # (... M D D)
                vobs = B.sum(B.abs(obs_diff) ** p, axis=-3) / M  # (... D D)
                return B.sum((vobs - vfcts) ** 2, axis=(-2, -1))  # (...)
            
            forecasts, _ = sr.core.utils.multivariate_array_check(forecasts, np.zeros_like(observations.mean(1)), 1, group_on_dim, backend="numpy")
            observations, _ = sr.core.utils.multivariate_array_check(observations, np.zeros_like(observations.mean(1)), 1, group_on_dim, backend="numpy")

            reference_score = ref_variogram_score_not_target_each_ens_indep(forecasts, observations, backend="numpy", p=beta_np.item())

        elif target_each_ens_indep:
            assert nens_input == nens_target, "Ensemble sizes must be equal"
            ref_scores = [ ]
            for i in range(nens_input):
                forecasts_reshaped, observations_reshaped = sr.core.utils.multivariate_array_check( copy.deepcopy(forecasts[:, i:i+1]), copy.deepcopy(observations[:, i]) , 1, group_on_dim, backend="numpy")
                _ = sr.core.variogram._score.variogram_score(forecasts_reshaped, observations_reshaped, backend="numpy", p=beta_np.item())
                ref_scores.append(_)
            # Take the mean of the scores matrices across the 1st dimension (w/ zero indexing)
            reference_score = np.mean(np.stack(ref_scores, axis=0),axis=0)
        
        if group_on_dim == -1:
            reference_score = reference_score * (node_weights_np/node_weights_np.sum())            
        elif group_on_dim == -2:
            reference_score = reference_score * (feature_weights_np/feature_weights_np.size)
        elif group_on_dim in [-3, -4]:    
            reference_score = reference_score * (feature_weights_np/feature_weights_np.size)
            reference_score = reference_score * (node_weights_np[..., None]/node_weights_np.sum())
        
        reference_score = reference_score.mean(0)

        np.testing.assert_allclose(score, reference_score, rtol=1e-3)
    
    @pytest.mark.parametrize(
        "implementation",
        ["vectorized", "low_mem"],
    )
    def test_kcrps(self, implementation: str) -> None:
        bs, nens, timesteps, latlon, nvars = 10, 5, 4, 2, 3

        preds = torch.randn(bs, nens, timesteps, latlon, nvars, dtype=torch.float32)
        target = torch.randn(bs, timesteps, latlon, nvars, dtype=torch.float32)
        node_weights = torch.ones(latlon, dtype=torch.float32)
        feature_weights = torch.ones(nvars, dtype=torch.float32)

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()


        dtype = torch.float32
        preds_, target_, node_weights_, feature_weights_ = (
            preds.to(dtype),
            target.to(dtype),
            node_weights.to(dtype),
            feature_weights.to(dtype),
        )

        kcrps_score = kCRPS(
            node_weights=node_weights_,
            feature_weights=feature_weights_,
            fair=True,
            implementation=implementation,
        )

        score = kcrps_score.forward(
            preds_,
            target_,
            squash=False,
            feature_scaling=True,
            feature_indices=None,
        )
        score = score.to(torch.float32).numpy()


        # Use NumPy arrays for reference calculation
        reference_score = sr.crps_ensemble(preds_np, target_np, axis=1, estimator="fair", backend="numpy")
        reference_score = reference_score * (node_weights_np[..., None]/node_weights_np.sum())
        reference_score = reference_score * (feature_weights_np/feature_weights_np.size)
        reference_score = reference_score.mean(0)

        np.testing.assert_allclose(score, reference_score, rtol=1e-1)

    @pytest.mark.parametrize(
        "losses_weights",
        [
            [0.4, 0.6, 0.2],
            [0.6,0.4, 0.1],
            [0.69, 0.3, 0.01],

        ],
    )
    def test_composite_loss(self, losses_weights: list) -> None:

        bs, nens_input, timesteps, latlon, nvars = 2, 8, 6, 7, 5
        nens_target = 5
        preds = torch.randn(bs, nens_input, timesteps, latlon, nvars)
        target = torch.randn(bs, nens_target, timesteps, latlon, nvars)
        feature_weights = torch.ones(nvars)
        node_weights = torch.ones(latlon)
        beta = torch.as_tensor(1.0)

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()
        beta_np = beta.numpy()


        losses = [
            DictConfig({"_target_": "anemoi.training.losses.ignorance.IgnoranceScore", "eps": self.eps}),
            DictConfig({"_target_": "anemoi.training.losses.mse.WeightedMSELoss", }),
            DictConfig({"_target_": "anemoi.training.losses.energy.EnergyScore", "group_on_dim": -1})
        ]


        composite_loss = CompositeLoss(
            losses=losses,
            loss_weights=losses_weights,
            node_weights=node_weights,
            feature_weights=feature_weights,
        )

        score = composite_loss.forward(
            preds,
            target,
            feature_scaling=True,
            feature_indices=None,
            squash=True
        )
        score = score.numpy()

        # Test individual losses
        ignorance_loss = hydra.utils.instantiate(losses[0], node_weights=node_weights, feature_weights=feature_weights)
        mse_loss = hydra.utils.instantiate(losses[1], node_weights=node_weights, feature_weights=feature_weights)
        energy_loss = hydra.utils.instantiate(losses[2], node_weights=node_weights, feature_weights=feature_weights)

        ignorance_score = ignorance_loss(preds, target, feature_scaling=True, feature_indices=None, squash=True)
        mse_score = mse_loss(preds, target, feature_scaling=True, feature_indices=None, squash=True)
        energy_score = energy_loss(preds, target, feature_scaling=True, feature_indices=None, squash=True)

        # Calculate expected composite score
        expected_score = (
            losses_weights[0] * ignorance_score +
            losses_weights[1] * mse_score +
            losses_weights[2] * energy_score
        )

        # Compare composite score with expected score
        np.testing.assert_allclose(score, expected_score.numpy(), rtol=1e-5)

        # Test that changing weights affects the score
        composite_loss_2 = CompositeLoss(
            losses=losses,
            loss_weights=[0.5, 0.3, 0.2],  # Different weights
            node_weights=node_weights,
            feature_weights=feature_weights,
        )

        score_2 = composite_loss_2.forward(
            preds,
            target,
            feature_scaling=True,
            feature_indices=None,
            squash=True
        )

        # Ensure the scores are different
        assert not np.allclose(score, score_2.numpy())


    def test_mse_loss(self) -> None:
        bs, nens, timesteps, latlon, nvars = 2, 4, 6, 3, 3
        preds = torch.randn(bs, nens, timesteps, latlon, nvars)
        target = torch.randn(bs, timesteps, latlon, nvars)
        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars) + 0.01

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()

        mse_loss = WeightedMSELoss(node_weights=node_weights, feature_weights=feature_weights)
        loss = mse_loss(preds.clone(), target.clone(), squash=False)

        # Reference calculation
        mse = (preds_np.mean(1) - target_np) ** 2
        weighted_mse = mse * (node_weights_np[:, None]/node_weights_np.sum()) * (feature_weights_np/feature_weights_np.size)
        reference_loss = weighted_mse.mean(0) 

        np.testing.assert_allclose(loss, reference_loss, rtol=1e-5)

    def test_mae_loss(self) -> None:
        bs, nens, timesteps, latlon, nvars = 2, 4, 6, 3 , 3
        preds = torch.randn(bs, nens, timesteps, latlon, nvars)
        target = torch.randn(bs, timesteps, latlon, nvars)
        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars) + 0.01

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()

        mae_loss = WeightedMAELoss(node_weights=node_weights, feature_weights=feature_weights)
        loss = mae_loss(preds.clone(), target.clone(), squash=False)

        # Reference calculation
        mae = np.abs(preds_np.mean(1) - target_np)
        weighted_mae = mae * (node_weights_np[:, None]/node_weights_np.sum()) * (feature_weights_np/feature_weights_np.size)
        reference_loss = weighted_mae.mean(0)

        np.testing.assert_allclose(loss, reference_loss, rtol=1e-5)

    def test_vae_loss(self) -> None:
        #TODO(rilwan-ade): Add a test for unsquashed VAE loss e.g
        bs, inp_ens_size, timesteps, latlon, nvars = 2, 6, 3, 4, 5
        timesteps_latent, latlon_latent, latent_dim = 7, 5, 5
        x_rec = torch.randn(bs, inp_ens_size, timesteps, latlon, nvars)
        x_target = x_rec + torch.randn(bs, inp_ens_size, timesteps, latlon, nvars) * 0.05  # Add small noise

        z_mu = torch.randn(bs, timesteps, latlon_latent, latent_dim)
        z_logvar = torch.randn(bs, timesteps, latlon_latent, latent_dim)
        divergence_loss_weight = 0.1

        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars) + 0.01
        latent_node_weights = torch.rand(latlon_latent)

        # creating the numpy versions of the tensors
        x_rec_np = x_rec.numpy()
        x_target_np = x_target.numpy()
        z_mu_np = z_mu.numpy()
        z_logvar_np = z_logvar.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()
        latent_node_weights_np = latent_node_weights.numpy()

        reconstruction_loss = WeightedMSELoss(node_weights=node_weights, feature_weights=feature_weights)
        divergence_loss = KLDivergenceLoss(node_weights=latent_node_weights)
        vae_loss = VAELoss(
            node_weights=node_weights,
            feature_weights=feature_weights,
            reconstruction_loss=reconstruction_loss,
            divergence_loss=divergence_loss,
            latent_node_weights=latent_node_weights,
            divergence_loss_weight=divergence_loss_weight,
        )

        loss = vae_loss(x_rec, x_target, z_mu=z_mu, z_logvar=z_logvar, squash=True )

        # Reference calculation (squashed)
        mse = (x_rec_np - x_target_np) ** 2
        weighted_mse = mse * (node_weights_np[:, None]/node_weights_np.sum()) * (feature_weights_np/feature_weights_np.size)
        rec_loss_ref = weighted_mse.mean((0,1)).sum()

        kl_div = -0.5 * (1 + z_logvar_np - z_mu_np**2 - np.exp(z_logvar_np))
        weighted_kl_div = kl_div * (latent_node_weights_np[:, None]/latent_node_weights_np.sum())
        div_loss_ref = weighted_kl_div.mean(0).sum()

        vae_loss_ref = rec_loss_ref + divergence_loss_weight * div_loss_ref

        assert isinstance(loss, dict)
        assert vae_loss.name in loss
        assert reconstruction_loss.name in loss
        assert divergence_loss.name in loss

        # Check if the calculated losses are close to the reference calculations
        assert np.allclose(loss[reconstruction_loss.name], rec_loss_ref, rtol=1e-5)
        assert np.allclose(loss[divergence_loss.name], div_loss_ref, rtol=1e-5)
        assert np.allclose(loss[vae_loss.name], vae_loss_ref, rtol=1e-3)


    def test_renyi_divergence_loss(self) -> None:
        bs, inp_ens_size, timesteps, latlon, latent_dim = 2, 4, 3, 4, 7
        mu = torch.randn(bs, inp_ens_size, timesteps, latlon, latent_dim)
        logvar = torch.randn(bs, inp_ens_size, timesteps, latlon, latent_dim)
        node_weights = torch.rand(latlon)
        alpha = 2.0

        renyi_loss = RenyiDivergenceLoss(alpha=alpha, node_weights=node_weights, feature_weights=torch.ones(latent_dim))
        loss = renyi_loss(mu.clone(), logvar.clone(), squash=False, feature_scaling=False)

        # Reference calculation
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        renyi_div = (1 / (alpha - 1)) * torch.log(1 + (alpha - 1) * kl_div)
        weighted_renyi_div = renyi_div * node_weights[..., None] / node_weights.sum()
        reference_loss = weighted_renyi_div.mean(0) 

        assert torch.allclose(loss, reference_loss, rtol=1e-5)

    @pytest.mark.parametrize("spectral_loss_method, nens_inp, nens_target, target_each_ens_indep", [("2D_spatial", 1, 2, False),
                                                                                                    ("2D_spatial", 3, 3, True), ("1D_temporal", 1, 1, False), 
                                                                                                    ("1D_temporal", 3, 3, True), ("3D_spatiotemporal", 2, 3, False),
                                                                                                    ("3D_spatiotemporal", 3, 3, True),
                                                                                                    ])
    #TODO(rilwan-ade): Add tests for SHTAmplitudePhaseLoss and SHTComplexMSELoss
    @pytest.mark.skip(reason="Temporarily disabled")
    def test_spectral_energy_loss(self, spectral_loss_method, nens_inp, nens_target, target_each_ens_indep) -> None:
        bs, timesteps, nlat, nlon, nvars = 2, 3, 4, 5, 6
        nens_inp = 4
        nens_target = 5
        latlon = nlat * nlon
        preds = torch.randn(bs, nens_inp, timesteps, latlon, nvars)
        target = torch.randn(bs, nens_target, timesteps, latlon, nvars)
        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars) + 0.01

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()

        spectral_loss = SpectralEnergyLoss(
            node_weights=node_weights,
            nlat=nlat,
            nlon=nlon,
            feature_weights=feature_weights,
            beta=2,
            spectral_loss_method=spectral_loss_method,
            target_each_ens_indep=target_each_ens_indep,
        )
        loss = spectral_loss(preds, target, squash=False)

        # assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Reference calculation

        if target_each_ens_indep:
            preds = einops.rearrange(preds, "bs ens timesteps latlon nvars -> (bs ens) timesteps latlon nvars")
            target = einops.rearrange(target, "bs ens timesteps latlon nvars -> (bs ens) timesteps latlon nvars")

        # Use NumPy arrays for reference calculation
        if spectral_loss_method == "2D_spatial":
            sht = th.RealSHT(nlat, nlon, grid="legendre-gauss")

            with torch.autocast(enabled=False):
                preds_sht = sht.grid2spec(preds_np.reshape(-1, nlat, nlon, nvars))
                target_sht = sht.grid2spec(target_np.reshape(-1, nlat, nlon, nvars))

            preds_energy = np.abs(preds_sht)
            target_energy = np.abs(target_sht)
        elif spectral_loss_method == "1D_temporal":
            preds_fft = np.abs(fftn(preds_np, axes=2))
            target_fft = np.abs(fftn(target_np, axes=1))
            preds_energy = preds_fft
            target_energy = target_fft
        else:  # 3D_spatiotemporal
            sht = th.RealSHT(nlat, nlon, grid="legendre-gauss")
            with torch.autocast(enabled=False):
                preds_sht = sht.grid2spec(preds_np.reshape(-1, nlat, nlon, nvars))
                target_sht = sht.grid2spec(target_np.reshape(-1, nlat, nlon, nvars))
            preds_fft = np.abs(fftn(preds_sht, axes=2))
            target_fft = np.abs(fftn(target_sht, axes=1))
            preds_energy = preds_fft
            target_energy = target_fft

        energy_diff = preds_energy - target_energy
        ref_loss = (np.abs(energy_diff) ** 2) * (node_weights_np[:, None] / node_weights_np.sum()) * feature_weights_np
        ref_loss = ref_loss.mean(0)

        if target_each_ens_indep:
            ref_loss = ref_loss * nens_inp

        assert np.allclose(loss.item(), ref_loss, rtol=1e-5)

    @pytest.mark.parametrize("group_on_dim, implementation", [
        (-1, "vectorized"),
        (-1, "low_mem"),
        (-2, "vectorized"),
        (-3, "vectorized")
    ])
    def test_multivariate_kcrps(self, group_on_dim: int, implementation: str):
        bs, nens, timesteps, latlon, nvars = 2, 4, 3, 6, 3
        preds = torch.randn(bs, nens, timesteps, latlon, nvars)
        target = torch.randn(bs, timesteps, latlon, nvars)
        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars)

        # Create MultivariatekCRPS instance
        mkcrps = MultivariatekCRPS(
            node_weights=node_weights,
            feature_weights=feature_weights,
            group_on_dim=group_on_dim,
            beta=1.0,
            p_norm=1.5,
            fair=True,
            implementation=implementation
        )

        # Calculate loss
        loss = mkcrps(preds, target)

        # Basic checks
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Check if loss changes with different inputs
        loss2 = mkcrps(preds * 1.1, target)
        assert loss != loss2

        # Check if loss is non-negative
        assert loss >= 0

        # TODO: Add more specific checks based on the expected behavior for each group_on_dim

    @pytest.mark.parametrize(
        "patch_method, group_on_dim, param_groups, data_indices_model_output",
        [
            (
                "group_by_variable",
                (-1),
                {"q": [0, 2, 4], "t": [1, 3, 5], "v": [6, 8, 11], "z": [7], "u": [9, 10]},
                {
                    "name_to_index": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "z43": 7,
                        "15v": 8,
                        "10u": 9,
                        "u_500": 10,
                        "v_900": 11,
                    },
                    "full_idxs": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "z43": 7,
                        "15v": 8,
                        "10u": 9,
                        "u_500": 10,
                        "v_900": 11,
                    },
                },
            ),
            (
                "group_by_pressurelevel",
                (-1),
                {"sfc": [0, 1, 6, 7], "500": [2, 3], "850": [4, 5]},
                {
                    "name_to_index": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "10u": 7,
                    },
                    "full_idxs": {
                        "20q": 0,
                        "t25": 1,
                        "q_500": 2,
                        "t_500": 3,
                        "q_850": 4,
                        "t_850": 5,
                        "v10": 6,
                        "10u": 7,
                    },
                },
            ),
        ],
    )  
    def test_grouped_multivariate_kcrps(self, patch_method, group_on_dim, param_groups, data_indices_model_output) -> None:
        bs, nens, timesteps, latlon, nvars = 2, 4, 3, 6, len(data_indices_model_output["full_idxs"])
        preds = torch.randn(bs, nens, timesteps, latlon, nvars)
        target = torch.randn(bs, timesteps, latlon, nvars)
        node_weights = 0.5 + torch.rand(latlon, dtype=torch.float32)
        feature_weights = 0.5 + torch.rand(nvars, dtype=torch.float32)
        beta = 1.0

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()
        beta_np = np.asarray(beta)

        data_indices_model_output = OmegaConf.create(data_indices_model_output)
        # Create GroupedMultivariatekCRPS instance
        gmkcrps = GroupedMultivariatekCRPS(
            node_weights=node_weights,
            feature_weights=feature_weights,
            patch_method=patch_method,
            group_on_dim=group_on_dim,
            beta=1.0,
            p_norm=1.5,
            fair=True,
            data_indices_model_output=data_indices_model_output,
            implementation="vectorized"
        )

        # Calculate loss
        loss = gmkcrps(preds, target, squash=True)

        # Basic checks
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Check if loss changes with different inputs
        loss2 = gmkcrps(preds * 1.1, target)
        assert loss != loss2

        # Check if loss is non-negative
        assert loss >= 0

        

        # Test with squash=False
        unsquashed_loss = gmkcrps(preds, target, squash=False)
        if group_on_dim == -2:
            expected_shape = (timesteps, nvars,)
        elif group_on_dim == -3:
            expected_shape = (latlon,nvars)
        elif group_on_dim == -1:
            expected_shape = (timesteps, latlon,)
        assert unsquashed_loss.shape == expected_shape

    def test_spread_skill_loss(self):
        bs, nens, timesteps, latlon, nvars = 2, 5, 3, 4, 2
        preds = torch.randn(bs, nens, timesteps, latlon, nvars)
        target = torch.randn(bs, timesteps, latlon, nvars)
        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars)

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        target_np = target.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()

        spread_skill_loss = SpreadSkillLoss(node_weights=node_weights, feature_weights=feature_weights)
        loss = spread_skill_loss(preds, target, squash=True)

        # Reference calculation
        rmse = np.sqrt(np.mean((preds_np.mean(axis=1) - target_np) ** 2, axis=0))
        ens_stdev = np.sqrt(np.mean(np.var(preds_np, axis=1, ddof=1), axis=0))
        
        weighted_rmse = rmse * (node_weights_np[:, None] / node_weights_np.sum()) * feature_weights_np
        weighted_ens_stdev = ens_stdev * (node_weights_np[:, None] / node_weights_np.sum()) * feature_weights_np
        
        reference_loss = np.mean(weighted_ens_stdev / weighted_rmse)

        assert np.allclose(loss.item(), reference_loss, rtol=1e-5)

    @pytest.mark.parametrize("beta", [1, 2, 3])
    def test_spread_loss(self, beta):
        bs, nens, timesteps, latlon, nvars = 2, 5, 3, 4, 2
        preds = torch.randn(bs, nens, timesteps, latlon, nvars)
        target = torch.randn(bs, timesteps, latlon, nvars)  # Note: target is not used in SpreadLoss
        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars)

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()

        spread_loss = SpreadLoss(node_weights=node_weights, feature_weights=feature_weights, beta=beta)
        loss = spread_loss(preds, target, squash=True)

        # Reference calculation
        spread = np.power(
            np.mean(np.power(preds_np - np.mean(preds_np, axis=1, keepdims=True), beta), axis=1),
            1 / beta
        )
        weighted_spread = spread * (node_weights_np[:, None] / node_weights_np.sum()) * feature_weights_np
        reference_loss = np.mean(weighted_spread)

        assert np.allclose(loss.item(), reference_loss, rtol=1e-5)

    def test_zero_spread_rate_loss(self):
        bs, nens, timesteps, latlon, nvars = 2, 5, 3, 4, 2
        preds = torch.randn(bs, nens, timesteps, latlon, nvars)
        target = torch.randn(bs, timesteps, latlon, nvars)  # Note: target is not used in ZeroSpreadRateLoss
        node_weights = torch.rand(latlon)
        feature_weights = torch.rand(nvars)

        # Create NumPy versions of the tensors
        preds_np = preds.numpy()
        node_weights_np = node_weights.numpy()
        feature_weights_np = feature_weights.numpy()

        zero_spread_rate_loss = ZeroSpreadRateLoss(node_weights=node_weights, feature_weights=feature_weights)
        loss = zero_spread_rate_loss(preds, target, squash=True)

        # Reference calculation
        spread = np.mean((preds_np - np.mean(preds_np, axis=1, keepdims=True)) ** 2, axis=1)
        spread_occurrence = (spread == 0).astype(float)
        weighted_spread_occurrence = spread_occurrence * (node_weights_np[:, None] / node_weights_np.sum()) * feature_weights_np
        reference_loss = np.mean(weighted_spread_occurrence)

        assert np.allclose(loss.item(), reference_loss, rtol=1e-5)

