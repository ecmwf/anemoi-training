import pytest
import torch
from omegaconf import OmegaConf
from anemoi.training.data.datamodule import (
    AnemoiDatasetsDataModule,
    AnemoiForecastingDataModule,
    AnemoiEnsForecastingDataModule,
    AnemoiReconstructionDataModule,
    AnemoiEnsReconstructionDataModule
)
from anemoi.training.data.dataset import NativeGridDataset, EnsNativeGridDataset
from unittest.mock import MagicMock, patch
from anemoi.models.data_indices.collection import IndexCollection
from test_dataset import MockDataReader

from torch.utils.data import DataLoader
@pytest.fixture
def mock_config():
    config = {
        "data": {
            "frequency": "6h",
            "timestep": "6h",
            "resolution": "1deg"
        },
        "hardware": {
            "num_gpus_per_node": 4,
            "num_nodes": 1,
            "num_gpus_per_model": 1,
            "num_gpus_per_ensemble": 1,
            "paths": {
                "checkpoints": "/path/to/checkpoints",
                "plots": "/path/to/plots"
            }
        },
        "training": {
            "rollout": {
                "start": 1,
                "max": 12,
                "epoch_increment": 1
            },
            "multistep_input": 2,
            "ic_ensemble_size": 5,
            "noise_sample_per_ic": 1,
            "prediction_strategy": "tendency",
            "nens_per_device": 5
        },
        "dataloader": {
            "batch_size": {
                "training": 4,
                "validation": 8,
                "test": 8
            },
            "num_workers": {
                "training": 4,
                "validation": 2,
                "test": 2
            },
            "prefetch_factor": 2,
            "limit_batches": {
                "training": 1.0
            },
            "training": {
                "start": "2010-01-01",
                "end": "2015-12-31"
            },
            "validation": {
                "start": "2016-01-01",
                "end": "2017-12-31"
            },
            "test": {
                "start": "2018-01-01",
                "end": "2019-12-31"
            },
            "predict": {
                "start": "2020-01-01",
                "end": "2020-12-31"
            }
        },
        "diagnostics": {
            "eval": {
                "enabled": True,
                "rollout": 48
            }
        }
    }
    return OmegaConf.create(config)

@pytest.fixture
def mock_open_dataset():
    def _mock_open_dataset(config):
        return MagicMock()
    return _mock_open_dataset

@pytest.fixture
def mock_native_grid_dataset():
    dataset = MagicMock(spec=NativeGridDataset)
    dataset.statistics = {"mean": 0, "std": 1}
    dataset.statistics_tendencies = {"mean": 0, "std": 1}
    dataset.metadata = {"some_key": "some_value"}
    dataset.name_to_index = {"var1": 0, "var2": 1}
    dataset.resolution = "1deg"
    
    # Create a MockDataReader instance
    mock_data_reader = MockDataReader(
        shape=(200, 10, 1, 1000),  # (time, variables, ensemble, gridpoints)
        statistics={"mean": 0, "std": 1},
        metadata={"some_key": "some_value"},
        name_to_index={"var1": 0, "var2": 1},
        resolution="1deg"
    )
    
    # Set the data property to the MockDataReader instance
    dataset.data = mock_data_reader
    
    return dataset

@pytest.fixture
def mock_ens_native_grid_dataset():
    dataset = MagicMock(spec=EnsNativeGridDataset)
    dataset.statistics = {"mean": 0, "std": 1}
    dataset.statistics_tendencies = {"mean": 0, "std": 1}
    dataset.metadata = {"some_key": "some_value"}
    dataset.name_to_index = {"var1": 0, "var2": 1}
    dataset.resolution = "1deg"
    
    # Create a MockDataReader instance with ensemble size > 1
    mock_data_reader = MockDataReader(
        shape=(200, 10, 5, 1000),  # (time, variables, ensemble, gridpoints)
        statistics={"mean": 0, "std": 1},
        metadata={"some_key": "some_value"},
        name_to_index={"var1": 0, "var2": 1},
        resolution="1deg"
    )
    
    # Set the data property to the MockDataReader instance
    dataset.data = mock_data_reader
    
    # Set additional properties specific to EnsNativeGridDataset
    dataset.ensemble_size = 4  # Ensemble size minus 1 (for analysis)
    dataset.num_eda_members = 4
    dataset.eda_flag = True
    dataset.num_analysis_members = 1
    dataset.ens_members_per_device = 4
    dataset.num_gpus_per_ens = 1
    dataset.num_gpus_per_model = 1
    
    return dataset


def test_anemoi_forecasting_datamodule(mock_config, mock_open_dataset, mock_native_grid_dataset):
    with patch("anemoi.training.data.datamodule.open_dataset", mock_open_dataset):
        datamodule = AnemoiForecastingDataModule(mock_config)
        
        assert datamodule.rollout == 12

        with patch.object(datamodule, "_get_dataset", return_value=mock_native_grid_dataset):
            train_dataset = datamodule.ds_train
            valid_dataset = datamodule.ds_valid
            test_dataset = datamodule.ds_test
            predict_dataset = datamodule.ds_predict

            assert isinstance(train_dataset, NativeGridDataset)
            assert isinstance(valid_dataset, NativeGridDataset)
            assert isinstance(test_dataset, NativeGridDataset)
            assert isinstance(predict_dataset, NativeGridDataset)

        assert datamodule.statistics == {"mean": 0, "std": 1}
        assert datamodule.statistics_tendencies == {"mean": 0, "std": 1}
        assert datamodule.metadata == {"some_key": "some_value"}

        # Test _get_dataloader
        dataloader = datamodule._get_dataloader(mock_native_grid_dataset, "training")
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 4
        assert dataloader.num_workers == 4


def test_anemoi_ens_forecasting_datamodule(mock_config, mock_open_dataset, mock_ens_native_grid_dataset):
    with patch("anemoi.training.data.datamodule.open_dataset", mock_open_dataset):
        datamodule = AnemoiEnsForecastingDataModule(mock_config)
        
        assert datamodule.nens_per_device == 5
        assert datamodule.ens_comm_group_id == 0
        assert datamodule.ens_comm_group_rank == 0
        assert datamodule.ens_comm_num_groups == 4

        assert datamodule.rollout == 12

        with patch.object(datamodule, "_get_dataset", return_value=mock_ens_native_grid_dataset):
            train_dataset = datamodule.ds_train
            valid_dataset = datamodule.ds_valid
            test_dataset = datamodule.ds_test
            predict_dataset = datamodule.ds_predict

            assert isinstance(train_dataset, EnsNativeGridDataset)
            assert isinstance(valid_dataset, EnsNativeGridDataset)
            assert isinstance(test_dataset, EnsNativeGridDataset)
            assert isinstance(predict_dataset, EnsNativeGridDataset)

        assert datamodule.statistics == {"mean": 0, "std": 1}
        assert datamodule.statistics_tendencies == {"mean": 0, "std": 1}
        assert datamodule.metadata == {"some_key": "some_value"}

        # Test _get_dataloader
        dataloader = datamodule._get_dataloader(mock_native_grid_dataset, "training")
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 4
        assert dataloader.num_workers == 4

def test_anemoi_reconstruction_datamodule(mock_config, mock_open_dataset, mock_native_grid_dataset):
    with patch("anemoi.training.data.datamodule.open_dataset", mock_open_dataset):
        if hasattr(mock_config.training, 'rollout'):
            delattr(mock_config.training, 'rollout')
        datamodule = AnemoiReconstructionDataModule(mock_config)

        

        with patch.object(datamodule, "_get_dataset", return_value=mock_native_grid_dataset):
            train_dataset = datamodule.ds_train
            valid_dataset = datamodule.ds_valid
            test_dataset = datamodule.ds_test

            assert isinstance(train_dataset, NativeGridDataset)
            assert isinstance(valid_dataset, NativeGridDataset)
            assert isinstance(test_dataset, NativeGridDataset)



def test_anemoi_ens_reconstruction_datamodule(mock_config, mock_open_dataset, mock_ens_native_grid_dataset):
    with patch("anemoi.training.data.datamodule.open_dataset", mock_open_dataset):
        if hasattr(mock_config.training, 'rollout'):
            delattr(mock_config.training, 'rollout')
        datamodule = AnemoiEnsReconstructionDataModule(mock_config)

        assert datamodule.nens_per_device == 5
        assert datamodule.ens_comm_group_id == 0
        assert datamodule.ens_comm_group_rank == 0
        assert datamodule.ens_comm_num_groups == 4


        with patch.object(datamodule, "_get_dataset", return_value=mock_ens_native_grid_dataset):
            train_dataset = datamodule.ds_train
            valid_dataset = datamodule.ds_valid
            test_dataset = datamodule.ds_test

            assert isinstance(train_dataset, EnsNativeGridDataset)
            assert isinstance(valid_dataset, EnsNativeGridDataset)
            assert isinstance(test_dataset, EnsNativeGridDataset)

        assert datamodule.statistics == {"mean": 0, "std": 1}
        assert datamodule.statistics_tendencies == {"mean": 0, "std": 1}
        assert datamodule.metadata == {"some_key": "some_value"}

        # Test _get_dataloader
        dataloader = datamodule._get_dataloader(mock_native_grid_dataset, "training")
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 4
        assert dataloader.num_workers == 4
