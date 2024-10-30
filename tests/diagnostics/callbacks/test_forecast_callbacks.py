import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
from anemoi.training.diagnostics.callbacks.forecast import (
    RolloutEval,
    ForecastingLossBarPlot,
    PlotSample,
    AnemoiCheckpointRollout,
    EarlyStoppingRollout,
    RolloutScheduler,
    get_callbacks
)

@pytest.fixture
def mock_config():
    return OmegaConf.create({
        "diagnostics": {
            "metrics": {
                "rollout_eval": {
                    "rollout": 5,
                    "frequency": 1
                }
            },
            "eval": {
                "frequency": 1,
                "lead_time_to_eval": [1, 2, 3, 4, 5]
            },
            "plot": {
                "parameters": ["temperature", "pressure"],
                "sample_idx": 0,
                "per_sample": 1
            },
            "checkpoints": [
                {"type": "interval", "every_n_epochs": 1},
                {"type": "performance", "monitor": "val_loss", "mode": "min"}
            ],
            "early_stoppings": [
                {"monitor": "val_loss", "patience": 3, "mode": "min"}
            ]
        },
        "data": {
            "timestep": "6h",
            "diagnostic": ["humidity"]
        },
        "hardware": {
            "paths": {
                "checkpoints": "/tmp/checkpoints"
            }
        },
        "training": {
            "max_epochs": 10,
            "swa": {
                "enabled": True,
                "lr": 0.01
            }
        }
    })

@pytest.fixture
def mock_pl_module():
    module = MagicMock(spec=pl.LightningModule)
    module.data_indices = MagicMock()
    module.data_indices.model.output.name_to_index = {
        "temperature": 0,
        "pressure": 1,
        "humidity": 2
    }
    module.latlons_data = torch.rand(100, 2)
    module.loss = MagicMock()
    module.loss.name = "mse"
    return module

@pytest.fixture
def mock_trainer():
    return MagicMock(spec=pl.Trainer)

def test_rollout_eval_init(mock_config):
    callbacks = [MagicMock(), MagicMock()]
    rollout_eval = RolloutEval(mock_config, val_dset_len=100, callbacks_validation_batch_end=callbacks, callbacks_validation_epoch_end=callbacks)
    
    assert rollout_eval.rollout == 5
    assert rollout_eval.frequency == 1
    assert rollout_eval.eval_frequency == 1
    assert rollout_eval.lead_time_to_eval == [1, 2, 3, 4, 5]
    assert rollout_eval.callbacks_validation_batch_end == callbacks
    assert rollout_eval.callbacks_validation_epoch_end == callbacks

@patch('torch.autocast')
def test_rollout_eval_on_validation_batch_end(mock_autocast, mock_config, mock_pl_module, mock_trainer):
    callbacks = [MagicMock(), MagicMock()]
    rollout_eval = RolloutEval(mock_config, val_dset_len=100, callbacks_validation_batch_end=callbacks, callbacks_validation_epoch_end=callbacks)
    
    batch = torch.rand(10, 5, 100, 3)  # (bs, time, grid, vars)
    outputs = {"y_pred": [torch.rand(10, 100, 3) for _ in range(5)], "y": [torch.rand(10, 100, 3) for _ in range(5)]}
    
    rollout_eval.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, batch, 0)
    
    mock_pl_module._step.assert_called_once()
    for callback in callbacks:
        callback.on_validation_batch_end.assert_called_once()

def test_forecasting_loss_bar_plot_init(mock_config):
    plot = ForecastingLossBarPlot(mock_config)
    
    assert plot.counter == 0
    assert isinstance(plot.dict_rstep_loss_map, dict)
    assert plot.time_step == "6h"
    assert plot.lead_time_to_eval == [1, 2, 3, 4, 5]

def test_forecasting_loss_bar_plot_accumulate(mock_config, mock_pl_module):
    plot = ForecastingLossBarPlot(mock_config)
    
    outputs = {
        "y_pred": [torch.rand(10, 100, 3) for _ in range(5)],
        "y": [torch.rand(10, 100, 3) for _ in range(5)]
    }
    batch = torch.rand(10, 5, 100, 3)
    
    plot.accumulate(mock_trainer, mock_pl_module, outputs, batch)
    
    assert len(plot.dict_rstep_loss_map) == 5
    for key in plot.dict_rstep_loss_map:
        assert plot.dict_rstep_loss_map[key].shape == (3,)

def test_plot_sample_init(mock_config):
    plot = PlotSample(mock_config)
    
    assert plot.sample_idx == 0
    assert plot.lead_time_to_eval == [1, 2, 3, 4, 5]

@patch('anemoi.training.diagnostics.plots.plots.plot_predicted_multilevel_flat_sample')
def test_plot_sample_plot(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = PlotSample(mock_config)
    
    outputs = {
        "y_pred": [torch.rand(10, 100, 3) for _ in range(5)],
        "y": [torch.rand(10, 100, 3) for _ in range(5)],
        "data": torch.rand(10, 5, 100, 3)
    }
    batch = torch.rand(10, 5, 100, 3)
    
    plot._plot(mock_trainer, mock_pl_module, outputs, batch, 0, 1)
    
    assert mock_plot.call_count == 5

def test_anemoi_checkpoint_rollout_init(mock_config):
    checkpoint = AnemoiCheckpointRollout(config=mock_config, monitor="val_loss")
    
    assert checkpoint.config == mock_config
    assert checkpoint.monitor == "val_loss"
    assert checkpoint.curr_rollout_steps is None

def test_early_stopping_rollout_init():
    early_stopping = EarlyStoppingRollout(monitor="val_loss", patience=3, mode="min", timestep="6h")
    
    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == 3
    assert early_stopping.mode == "min"
    assert early_stopping.timestep == "6h"
    assert early_stopping.curr_rollout_steps is None

def test_rollout_scheduler_init():
    schedule = {0: 1, 10: 2, 20: 3}
    scheduler = RolloutScheduler(schedule, increment_on="training_epoch")
    
    assert scheduler.schedule == schedule
    assert scheduler.increment_on == "training_epoch"
    assert scheduler.current_rollout is None

def test_rollout_scheduler_calculate_rollout():
    schedule = {0: 1, 10: 2, 20: 3}
    scheduler = RolloutScheduler(schedule, increment_on="training_epoch")
    
    assert scheduler.calculate_rollout(5) == 1
    assert scheduler.calculate_rollout(15) == 2
    assert scheduler.calculate_rollout(25) == 3

def test_get_callbacks(mock_config):
    monitored_metrics = ["val_loss"]
    callbacks = get_callbacks(mock_config, monitored_metrics, val_dset_len=100)
    
    callback_types = [type(cb) for cb in callbacks]
    assert AnemoiCheckpointRollout in callback_types
    assert EarlyStoppingRollout in callback_types
    assert RolloutEval in callback_types
    assert any(isinstance(cb, pl.callbacks.LearningRateMonitor) for cb in callbacks)
    assert any(isinstance(cb, pl.callbacks.StochasticWeightAveraging) for cb in callbacks)

# Add more tests as needed for other callbacks and edge cases