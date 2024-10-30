# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from _pytest.fixtures import SubRequest
from anemoi.models.data_indices.collection import IndexCollection
from omegaconf import DictConfig
import numpy as np
from unittest.mock import MagicMock
import unittest
from anemoi.training.lightning_module.forecasting import AnemoiLightningModule


@pytest.fixture
def fake_data(request: SubRequest) -> tuple[DictConfig, IndexCollection]:
    config = DictConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
            },
            "training": {
                "feature_weighting": {
                    "default": 1,
                    "sfc": {
                        "z": 0.1,
                        "other": 100,
                    },
                    "pl": {"y": 0.5},
                    "inverse_tendency_variance_scaling": request.param.get("use_inverse_scaling", False),
                },
                "metrics": ["other", "y_850"],
                "pressure_level_scaler": request.param.get("pressure_level_scaler", linear_scaler),
            },
        },
    )
    name_to_index = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return config, data_indices


linear_scaler = {
    "_target_": "anemoi.training.data.scaling.LinearPressureLevelScaler",
    "minimum": 0.0,
    "slope": 0.001,
}
relu_scaler = {
    "_target_": "anemoi.training.data.scaling.ReluPressureLevelScaler",
    "minimum": 0.2,
    "slope": 0.001,
}
constant_scaler = {
    "_target_": "anemoi.training.data.scaling.NoPressureLevelScaler",
    "minimum": 1.0,
    "slope": 0.0,
}
polynomial_scaler = {
    "_target_": "anemoi.training.data.scaling.PolynomialPressureLevelScaler",
    "minimum": 0.2,
    "slope": 0.001,
}

expected_linear_scaling = torch.Tensor(
    [
        50 / 1000 * 0.5,  # y_50
        500 / 1000 * 0.5,  # y_500
        850 / 1000 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
    ],
)
expected_relu_scaling = torch.Tensor(
    [
        0.2 * 0.5,  # y_50
        500 / 1000 * 0.5,  # y_500
        850 / 1000 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
    ],
)
expected_constant_scaling = torch.Tensor(
    [
        1 * 0.5,  # y_50
        1 * 0.5,  # y_500
        1 * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
    ],
)
expected_polynomial_scaling = torch.Tensor(
    [
        ((50 / 1000) ** 2 + 0.2) * 0.5,  # y_50
        ((500 / 1000) ** 2 + 0.2) * 0.5,  # y_500
        ((850 / 1000) ** 2 + 0.2) * 0.5,  # y_850
        1,  # q
        0.1,  # z
        100,  # other
    ],
)


@pytest.mark.parametrize(
    ("fake_data", "expected_scaling"),
    [
        ({"pressure_level_scaler": linear_scaler, "use_inverse_scaling": False}, expected_linear_scaling),
        ({"pressure_level_scaler": relu_scaler, "use_inverse_scaling": False}, expected_relu_scaling),
        ({"pressure_level_scaler": constant_scaler, "use_inverse_scaling": False}, expected_constant_scaling),
        ({"pressure_level_scaler": polynomial_scaler, "use_inverse_scaling": False}, expected_polynomial_scaling),
    ],
    indirect=["fake_data"],
)
def test_loss_scaling_vals(fake_data: tuple[DictConfig, IndexCollection], expected_scaling: torch.Tensor) -> None:
    config, data_indices = fake_data

    # Mock the entire AnemoiLightningModule class
    with unittest.mock.patch('anemoi.training.lightning_module.forecasting.AnemoiLightningModule') as MockAnemoiLightningModule:
        # Create an instance of the mocked class
        mock_lightning_module = MockAnemoiLightningModule.return_value
        
        # Mock the model attribute
        mock_lightning_module.model = MagicMock()
        
        # Set up the get_feature_weights method to use the actual implementation
        mock_lightning_module.get_feature_weights.side_effect = AnemoiLightningModule.get_feature_weights.__func__
        
        # Call get_feature_weights on the mocked instance
        loss_scaling = mock_lightning_module.get_feature_weights(mock_lightning_module, config, data_indices)

    assert torch.allclose(loss_scaling, expected_scaling)
    
@pytest.mark.parametrize(
    ("fake_data", "expected_scaling"),
    [
        ({"pressure_level_scaler": linear_scaler, "use_inverse_scaling": True}, expected_linear_scaling),
    ],
    indirect=["fake_data"],
)
def test_loss_scaling_with_inverse_variance(fake_data: tuple[DictConfig, IndexCollection], expected_scaling: torch.Tensor) -> None:
    config, data_indices = fake_data

    # Mock the entire AnemoiLightningModule class
    with unittest.mock.patch('anemoi.training.lightning_module.forecasting.AnemoiLightningModule') as MockAnemoiLightningModule:
        # Create an instance of the mocked class
        mock_lightning_module = MockAnemoiLightningModule.return_value
        
        # Mock the model attribute
        stdev = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        mock_lightning_module.model = MagicMock()
        mock_lightning_module.model.statistics_tendencies = {
            "stdev": stdev
        }
        
        # Set up the get_feature_weights method to use the actual implementation
        mock_lightning_module.get_feature_weights.side_effect = AnemoiLightningModule.get_feature_weights.__func__
        
        # Call get_feature_weights on the mocked instance
        loss_scaling = mock_lightning_module.get_feature_weights(mock_lightning_module, config, data_indices)

    # Calculate expected scaling with inverse variance
    expected_scaling = expected_scaling / (torch.tensor(stdev[data_indices.data.output.full]) ** 2)

    assert torch.allclose(loss_scaling, expected_scaling.to(loss_scaling.dtype), rtol=1e-5)

@pytest.mark.parametrize(
    "fake_data, metrics, expected_metric_range",
    [
        (
            linear_scaler,
            ["other", "y_850"],
            {
                "pl_y": [0, 1, 2],  # y_50, y_500, y_850
                "sfc": [3,4,5],
                "other": [5],
                "y_850": [2],
            },
        ),
        (
            linear_scaler,
            ["all"],
            {
                "pl_y": [0, 1, 2],  # y_50, y_500, y_850
                "sfc": [3,4,5],
                "q": [3],
                "z": [4],
                "y_50": [0],
                "y_500": [1],
                "y_850": [2],
                "other": [5],
                
            },
        ),
        (
            linear_scaler,
            ["group_all"],
            {
                "pl_y": [0, 1, 2],  # y_50, y_500, y_850
                "sfc": [3,4,5],
                "all": [0, 1, 2, 3, 4, 5, ],  # All indices
            },
        ),
        (
            linear_scaler,
            ["other", "y_850", "q_10"],
            {
                "pl_y": [0, 1, 2],  # y_50, y_500, y_850
                "sfc": [3,4,5],
                "other": [5],
                "y_850": [2]            },
        ),
    ],
    indirect=["fake_data"],
)
def test_metric_range(
    fake_data: tuple[DictConfig, IndexCollection],
    metrics: list[str],
    expected_metric_range: dict[str, list[int]],
) -> None:
    config, data_indices = fake_data
    config.training.metrics = metrics


    metric_range = AnemoiLightningModule.get_val_metric_ranges(config, data_indices)

    assert dict(metric_range) == expected_metric_range
