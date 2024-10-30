import pytest
import torch
import numpy as np
from omegaconf import OmegaConf
from anemoi.training.data.dataset import NativeGridDataset, EnsNativeGridDataset, worker_init_func
from unittest.mock import MagicMock, patch
from hypothesis import HealthCheck, given, settings, strategies as st


# Define strategies for our variables
st_rollout = st.integers(min_value=2, max_value=12)
st_multistep = st.integers(min_value=2, max_value=12)
st_ensemble_size = st.integers(min_value=2, max_value=8)
st_gridpoints = st.integers(min_value=12,max_value=24)
st_variable_count = st.integers(min_value=2, max_value=16)
st_time_increment = st.integers(min_value=2, max_value=2)

class MockDataReader:
    def __init__(self, shape=None, statistics=None, metadata=None, name_to_index=None, resolution=None):
        # Default values
        self.shape = shape or (200, 10, 1, 1000)  # (time, variables, ensemble, gridpoints)
        self.statistics = statistics or {"mean": 0, "std": 1}
        self._metadata = metadata or {"some_key": "some_value"}
        self.name_to_index = name_to_index or {"var1": 0, "var2": 1}
        self.resolution = resolution or "1deg"
        
        self.data = np.random.randn(*self.shape)
        self.statistics_tendencies = lambda x: {"mean": 0, "std": 1}

    def metadata(self):
        return self._metadata

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

@pytest.fixture(autouse=True)
def set_anemoi_base_seed(monkeypatch):
    monkeypatch.setenv("ANEMOI_BASE_SEED", "4")
    yield
    # The environment variable will be automatically reset after each test

@pytest.fixture
def mock_data_reader(request):
    # Check if custom parameters are provided
    params = getattr(request, 'param', {})
    
    # Create MockDataReader with custom or default values
    return MockDataReader(
        shape=params.get('shape'),
        statistics=params.get('statistics'),
        metadata=params.get('metadata'),
        name_to_index=params.get('name_to_index'),
        resolution=params.get('resolution')
    )

@pytest.fixture
def native_grid_dataset(mock_data_reader):
    return NativeGridDataset(
        data_reader=mock_data_reader,
        rollout=12,
        multistep=2,
        timeincrement=1,
        timestep="6h",
        model_comm_group_rank=0,
        model_comm_group_id=0,
        model_comm_num_groups=1,
        model_comm_group_nworkers=4,
        shuffle=True,
        label="test"
    )
 
@pytest.fixture
def ens_native_grid_dataset(request, mock_data_reader):
    ensemble_size = request.param
    
    # Update mock_data_reader shape for the chosen ensemble size
    mock_data_reader.shape = (200, 10, ensemble_size, 1000)
    
    return EnsNativeGridDataset(
        data_reader=mock_data_reader,
        rollout=12,
        multistep=2,
        timeincrement=1,
        comm_group_rank=0,
        comm_group_id=0,
        comm_num_groups=1,
        shuffle=True,
        label="test",
        ens_members_per_device=ensemble_size,
        num_gpus_per_ens=1,
        num_gpus_per_model=1
    )


@pytest.fixture
def reconstruction_native_grid_dataset(mock_data_reader):
    return NativeGridDataset(
        data_reader=mock_data_reader,
        rollout=0,
        multistep=12,
        timeincrement=1,
        timestep="6h",
        model_comm_group_rank=0,
        model_comm_group_id=0,
        model_comm_num_groups=1,
        model_comm_group_nworkers=4,
        shuffle=True,
        label="test"
    )

@pytest.fixture
def reconstruction_ens_native_grid_dataset(mock_data_reader):
    return EnsNativeGridDataset(
        data_reader=mock_data_reader,
        rollout=0,
        multistep=12,
        timeincrement=1,
        comm_group_rank=0,
        comm_group_id=0,
        comm_num_groups=1,
        shuffle=True,
        label="test",
        ens_members_per_device=1,
        num_gpus_per_ens=1,
        num_gpus_per_model=1
    )

def test_native_grid_dataset_init(native_grid_dataset):
    assert native_grid_dataset.rollout == 12
    assert native_grid_dataset.timeincrement == 1
    assert native_grid_dataset.timestep == "6h"
    assert native_grid_dataset.multi_step == 2
    assert native_grid_dataset.ensemble_size == 1

def test_native_grid_dataset_properties(native_grid_dataset):
    assert native_grid_dataset.statistics == {"mean": 0, "std": 1}
    assert native_grid_dataset.statistics_tendencies == {"mean": 0, "std": 1}
    assert native_grid_dataset.metadata == {"some_key": "some_value"}
    assert native_grid_dataset.name_to_index == {"var1": 0, "var2": 1}
    assert native_grid_dataset.resolution == "1deg"

def test_native_grid_dataset_per_worker_init(native_grid_dataset):
    native_grid_dataset.per_worker_init(n_workers=4, worker_id=0)
    assert native_grid_dataset.worker_id == 0
    assert native_grid_dataset.chunk_index_range is not None
    assert native_grid_dataset.rng is not None

@given(rollout=st_rollout, multistep=st_multistep, ensemble_size=st.just(1),
       gridpoints=st_gridpoints, variable_count=st_variable_count)
def test_native_grid_dataset_iter(rollout, multistep, ensemble_size, gridpoints, variable_count):
    shape = (100, variable_count, ensemble_size, gridpoints)
    mock_data_reader = MockDataReader(shape)
    
    dataset = NativeGridDataset(
        data_reader=mock_data_reader,
        rollout=rollout,
        multistep=multistep,
        timeincrement=1,
        timestep="6h",
        model_comm_group_rank=0,
        model_comm_group_id=0,
        model_comm_num_groups=1,
        model_comm_group_nworkers=4,
        shuffle=True,
        label="test"
    )
    
    dataset.per_worker_init(n_workers=4, worker_id=0)
    iterator = iter(dataset)
    first_item = next(iterator)
    assert isinstance(first_item, torch.Tensor)
    assert first_item.shape == (multistep+rollout, ensemble_size, gridpoints, variable_count)

def test_native_grid_dataset_len(native_grid_dataset):
    length = len(native_grid_dataset)
    assert isinstance(length, int)
    assert length > 0



@given(ensemble_size=st_ensemble_size.filter(lambda x: x > 1), time_increment=st_time_increment)
@settings(max_examples=1, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_ens_native_grid_dataset_with_hypothesis(ensemble_size, time_increment):
    # Manually instantiate MockDataReader with the generated ensemble_size
    mock_data_reader = MockDataReader(shape=(1000, 10, ensemble_size + 1, 25))
    
    dataset = EnsNativeGridDataset(
        data_reader=mock_data_reader,
        rollout=12,
        multistep=2,
        timeincrement=time_increment,
        comm_group_rank=0,
        comm_group_id=0,
        comm_num_groups=1,
        shuffle=True,
        label="test",
        ens_members_per_device=ensemble_size,
        num_gpus_per_ens=1,
        num_gpus_per_model=1
    )
    
    
    assert dataset.ensemble_size == ensemble_size
    assert dataset.num_eda_members == ensemble_size 
    assert dataset.eda_flag == True
    assert dataset.rollout == 12
    assert dataset.timeincrement == time_increment
    assert dataset.multi_step == 2
    assert dataset.ens_members_per_device == ensemble_size
    assert dataset.num_gpus_per_ens == 1
    assert dataset.num_gpus_per_model == 1

    dataset.per_worker_init(n_workers=4, worker_id=0)

    
    iterator = iter(dataset)
    first_item = next(iterator)
    assert isinstance(first_item, tuple)
    assert len(first_item) == 2 # output (analysis, eda)
    assert first_item[0].shape == (1, 2+12, 25, 10)
    assert first_item[1].shape == (ensemble_size, 2+12, 25, 10)

@given(rollout=st_rollout, multistep=st_multistep, ensemble_size=st_ensemble_size.filter(lambda x: x > 1),
       gridpoints=st_gridpoints, variable_count=st_variable_count)
def test_ens_native_grid_dataset_iter(rollout, multistep, ensemble_size, gridpoints, variable_count):
    shape = (200, variable_count, ensemble_size+1, gridpoints)
    mock_data_reader = MockDataReader(shape)
    
    dataset = EnsNativeGridDataset(
        data_reader=mock_data_reader,
        rollout=rollout,
        multistep=multistep,
        timeincrement=1,
        comm_group_rank=0,
        comm_group_id=0,
        comm_num_groups=1,
        shuffle=True,
        label="test",
        ens_members_per_device=ensemble_size,
        num_gpus_per_ens=1,
        num_gpus_per_model=1
    )
    
    dataset.per_worker_init(n_workers=4, worker_id=0)
    iterator = iter(dataset)
    first_item = next(iterator)
    assert isinstance(first_item, tuple)
    assert len(first_item) == 2 # output (analysis, eda)
    assert first_item[0].shape == (1, multistep+rollout, gridpoints, variable_count)
    assert first_item[1].shape == (ensemble_size, multistep+rollout, gridpoints, variable_count)


def test_native_grid_dataset_no_shuffle(mock_data_reader):
    dataset = NativeGridDataset(
        data_reader=mock_data_reader,
        rollout=12,
        multistep=2,
        timeincrement=1,
        timestep="6h",
        model_comm_group_rank=0,
        model_comm_group_id=0,
        model_comm_num_groups=1,
        model_comm_group_nworkers=4,
        shuffle=False,
        label="test"
    )
    dataset.per_worker_init(n_workers=4, worker_id=0)
    iterator1 = list(iter(dataset))
    iterator2 = list(iter(dataset))
    assert all(torch.allclose(a, b) for a, b in zip(iterator1, iterator2))




def test_native_grid_dataset_repr(native_grid_dataset):
    repr_str = repr(native_grid_dataset)
    assert "Dataset:" in repr_str
    assert "Rollout: 12" in repr_str
    assert "Multistep: 2" in repr_str
    assert "Timeincrement: 1" in repr_str


def test_reconstruction_native_grid_dataset_init(reconstruction_native_grid_dataset):
    assert reconstruction_native_grid_dataset.rollout == 0
    assert reconstruction_native_grid_dataset.multi_step == 12
    assert reconstruction_native_grid_dataset.timeincrement == 1
    assert reconstruction_native_grid_dataset.timestep == "6h"

@given(multistep=st_multistep, ensemble_size=st_ensemble_size,
       gridpoints=st_gridpoints, variable_count=st_variable_count)
def test_reconstruction_native_grid_dataset_iter(multistep, ensemble_size, gridpoints, variable_count):
    shape = (100, variable_count, ensemble_size, gridpoints)
    mock_data_reader = MockDataReader(shape)
    
    dataset = NativeGridDataset(
        data_reader=mock_data_reader,
        rollout=0,  # Always 0 for reconstruction
        multistep=multistep,
        timeincrement=1,
        timestep="6h",
        model_comm_group_rank=0,
        model_comm_group_id=0,
        model_comm_num_groups=1,
        model_comm_group_nworkers=4,
        shuffle=True,
        label="test"
    )
    
    dataset.per_worker_init(n_workers=4, worker_id=0)
    iterator = iter(dataset)
    first_item = next(iterator)
    assert isinstance(first_item, torch.Tensor)
    assert first_item.shape == (multistep, ensemble_size, gridpoints, variable_count)

def test_reconstruction_ens_native_grid_dataset_init(reconstruction_ens_native_grid_dataset):
    assert reconstruction_ens_native_grid_dataset.rollout == 0
    assert reconstruction_ens_native_grid_dataset.multi_step == 12
    assert reconstruction_ens_native_grid_dataset.timeincrement == 1

@given(multistep=st_multistep, ensemble_size=st_ensemble_size.filter(lambda x: x > 1),
       gridpoints=st_gridpoints, variable_count=st_variable_count)
@settings(max_examples=1)
def test_reconstruction_ens_native_grid_dataset_iter(multistep, ensemble_size, gridpoints, variable_count):
    shape = (100, variable_count, ensemble_size+1, gridpoints)
    mock_data_reader = MockDataReader(shape)
    
    dataset = EnsNativeGridDataset(
        data_reader=mock_data_reader,
        rollout=0,  # Always 0 for reconstruction
        multistep=multistep,
        timeincrement=1,
        comm_group_rank=0,
        comm_group_id=0,
        comm_num_groups=1,
        shuffle=True,
        label="test",
        ens_members_per_device=ensemble_size,
        num_gpus_per_ens=1,
        num_gpus_per_model=1
    )
    
    dataset.per_worker_init(n_workers=4, worker_id=0)
    iterator = iter(dataset)
    first_item = next(iterator)
    assert isinstance(first_item, tuple)
    assert len(first_item) == 2
    assert first_item[0].shape == (1, multistep, gridpoints, variable_count)
    assert first_item[1].shape == (ensemble_size, multistep, gridpoints, variable_count)

@given(ensemble_size=st_ensemble_size.filter(lambda x: x > 1), time_increment=st_time_increment, multistep=st_multistep)
@settings(max_examples=1, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_ens_native_grid_dataset_reconstruction_with_hypothesis(ensemble_size, time_increment, multistep):
    # Manually instantiate MockDataReader with the generated ensemble_size
    mock_data_reader = MockDataReader(shape=(1000, 10, ensemble_size + 1, 25))
    
    dataset = EnsNativeGridDataset(
        data_reader=mock_data_reader,
        rollout=0,  # Set to 0 for reconstruction
        multistep=multistep,
        timeincrement=time_increment,
        comm_group_rank=0,
        comm_group_id=0,
        comm_num_groups=1,
        shuffle=True,
        label="test",
        ens_members_per_device=ensemble_size,
        num_gpus_per_ens=1,
        num_gpus_per_model=1
    )
    
    assert dataset.ensemble_size == ensemble_size
    assert dataset.num_eda_members == ensemble_size 
    assert dataset.eda_flag == True
    assert dataset.rollout == 0  # Ensure rollout is 0 for reconstruction
    assert dataset.timeincrement == time_increment
    assert dataset.multi_step == multistep
    assert dataset.ens_members_per_device == ensemble_size
    assert dataset.num_gpus_per_ens == 1
    assert dataset.num_gpus_per_model == 1

    dataset.per_worker_init(n_workers=4, worker_id=0)
    
    iterator = iter(dataset)
    first_item = next(iterator)
    assert isinstance(first_item, tuple)
    assert len(first_item) == 2  # output (analysis, eda)
    assert first_item[0].shape == (1, multistep, 25, 10)
    assert first_item[1].shape == (ensemble_size, multistep, 25, 10)

@given(time_increment=st_time_increment, multistep=st_multistep)
@settings(max_examples=1, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_native_grid_dataset_reconstruction_with_hypothesis(time_increment, multistep):
    # Manually instantiate MockDataReader
    mock_data_reader = MockDataReader(shape=(1000, 10, 1, 25))
    
    dataset = NativeGridDataset(
        data_reader=mock_data_reader,
        rollout=0,  # Set to 0 for reconstruction
        multistep=multistep,
        timeincrement=time_increment,
        timestep="6h",
        model_comm_group_rank=0,
        model_comm_group_id=0,
        model_comm_num_groups=1,
        model_comm_group_nworkers=4,
        shuffle=True,
        label="test"
    )
    
    assert dataset.ensemble_size == 1
    assert dataset.rollout == 0  # Ensure rollout is 0 for reconstruction
    assert dataset.timeincrement == time_increment
    assert dataset.multi_step == multistep

    dataset.per_worker_init(n_workers=4, worker_id=0)
    
    iterator = iter(dataset)
    first_item = next(iterator)
    assert isinstance(first_item, torch.Tensor)
    assert first_item.shape == (multistep, 1, 25, 10)
