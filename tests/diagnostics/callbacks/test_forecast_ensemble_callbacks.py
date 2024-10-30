import pytest
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
from anemoi.training.diagnostics.callbacks.forecast_ensemble import (
    RolloutEvalEns,
    SpreadSkillPlot,
    RankHistogramPlot,
    PlotEnsembleInitialConditions,
    PlotEnsSample,
    PlotEnsPowerSpectrum,
    setup_rollout_eval_callbacks,
    get_callbacks
)

@pytest.fixture
def mock_config():
    return OmegaConf.create({
        "diagnostics": {
            "metrics": {
                "rollout_eval": {
                    "rollout": 5,
                    "frequency": 1,
                    "num_bins": 10
                }
            },
            "eval": {
                "frequency": 1,
                "lead_time_to_eval": [1, 2, 3, 4, 5]
            },
            "plot": {
                "parameters": ["temperature", "pressure"],
                "parameters_spectrum": ["temperature", "pressure"],
                "sample_idx": 0,
                "per_sample": 1,
                "ens_idx": [0, 1],
                "accumulation_levels_plot": [1, 2, 3],
                "cmap_accumulation": "viridis",
                "loss_map": True,
                "loss_bar": True,
                "plot_spectral_loss": True,
                "ens_sample": True,
                "rank_histogram": True,
                "spread_skill_plot": True,
                "ensemble_initial_conditions": True
            },
            "test": {
                "rollouteval": {
                    "eval": {"enabled": True, "frequency": 10, "rsteps_to_log": [1, 5, 10]},
                    "video": {"enabled": True, "frequency": 10, "rollout": 20},
                }
            }
        },
        "data": {
            "timestep": "6h",
            "diagnostic": ["humidity"]
        },
        "hardware": {
            "paths": {
                "plots": "/tmp/plots"
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
    module.data_indices.data.input.full = slice(None)
    module.data_indices.data.output.full = slice(None)
    module.latlons_data = torch.rand(100, 2)
    module.loss = MagicMock()
    module.loss.name = "mse"
    module.multi_step = 5
    module.rollout = 10
    module.ens_comm_group_rank = 0
    module.ens_comm_group_size = 2
    module.ens_comm_group = MagicMock()
    module._gather_matrix = torch.eye(2)
    return module

@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=pl.Trainer)
    trainer.logger = MagicMock()
    trainer.current_epoch = 0
    trainer.global_step = 0
    return trainer

def test_rollout_eval_ens_init(mock_config):
    callbacks = [MagicMock(), MagicMock()]
    rollout_eval = RolloutEvalEns(mock_config, val_dset_len=100, callbacks=callbacks)
    
    assert rollout_eval.rollout == 5
    assert rollout_eval.frequency == 1
    assert rollout_eval.eval_frequency == 1
    assert rollout_eval.lead_time_to_eval == [1, 2, 3, 4, 5]
    assert len(rollout_eval.callbacks_validation_batch_end) == 2
    assert len(rollout_eval.callbacks_validation_epoch_end) == 2

@patch('torch.autocast')
def test_rollout_eval_ens_on_validation_batch_end(mock_autocast, mock_config, mock_pl_module, mock_trainer):
    callbacks = [MagicMock(), MagicMock()]
    rollout_eval = RolloutEvalEns(mock_config, val_dset_len=100, callbacks=callbacks)
    
    batch = torch.rand(10, 5, 100, 3)  # (bs, time, grid, vars)
    outputs = {"y_pred": [torch.rand(10, 100, 3) for _ in range(5)], "y": [torch.rand(10, 100, 3) for _ in range(5)]}
    
    rollout_eval.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, batch, 0)
    
    mock_pl_module._step.assert_called_once()
    for callback in callbacks:
        callback.on_validation_batch_end.assert_called_once()

def test_spread_skill_plot_init(mock_config):
    plot = SpreadSkillPlot(mock_config, val_dset_len=100)
    
    assert plot.spread_skill.rollout == 5
    assert plot.spread_skill.nvar == 2
    assert plot.spread_skill.nbins == 10
    assert plot.spread_skill.time_step == 6
    assert plot.lead_time_to_eval == [1, 2, 3, 4, 5]

def test_spread_skill_plot_on_validation_batch_end(mock_config, mock_pl_module, mock_trainer):
    plot = SpreadSkillPlot(mock_config, val_dset_len=100)
    
    outputs = {
        "preds_denorm": [torch.rand(10, 100, 3) for _ in range(5)],
        "targets_denorm": [torch.rand(10, 100, 3) for _ in range(5)],
        "y_pred": [torch.rand(10, 100, 3) for _ in range(5)]
    }
    batch = torch.rand(10, 5, 100, 3)
    
    with patch.object(plot.spread_skill, 'calculate_spread_skill', return_value=(0, 0, 0, 0)):
        plot.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, batch, 0)
    
    assert plot.spread_skill.num_updates == 5

@patch('anemoi.training.diagnostics.callbacks.forecast_ensemble.plot_spread_skill')
@patch('anemoi.training.diagnostics.callbacks.forecast_ensemble.plot_spread_skill_bins')
def test_spread_skill_plot_on_validation_epoch_end(mock_plot_bins, mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = SpreadSkillPlot(mock_config, val_dset_len=100)
    plot.spread_skill.num_updates = 1
    plot.spread_skill.compute = MagicMock(return_value=(np.zeros(5), np.zeros(5), np.zeros((5, 10)), np.zeros((5, 10))))
    
    plot.on_validation_epoch_end(mock_trainer, mock_pl_module)
    
    mock_plot.assert_called_once()
    mock_plot_bins.assert_called_once()

def test_rank_histogram_plot_init(mock_config):
    plot = RankHistogramPlot(mock_config, ranks=MagicMock())
    
    assert plot.ranks is not None

@patch('anemoi.training.diagnostics.callbacks.forecast_ensemble.plot_rank_histograms')
def test_rank_histogram_plot_on_validation_epoch_end(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = RankHistogramPlot(mock_config, ranks=MagicMock())
    plot.ranks.ranks = torch.rand(10, 5, 3)
    
    plot.on_validation_epoch_end(mock_trainer, mock_pl_module)
    
    mock_plot.assert_called_once()

def test_plot_ensemble_initial_conditions_init(mock_config):
    plot = PlotEnsembleInitialConditions(mock_config)
    
    assert plot.sample_idx == 0

@patch('anemoi.training.diagnostics.callbacks.forecast_ensemble.plot_predicted_ensemble')
def test_plot_ensemble_initial_conditions_plot(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = PlotEnsembleInitialConditions(mock_config)
    plot.post_processors_state = MagicMock()
    plot.post_processors_state.processors.normalizer.inverse_transform.return_value = torch.rand(10, 5, 100, 3)
    
    ens_ic = torch.rand(10, 5, 100, 3)
    plot._plot(mock_trainer, mock_pl_module, ens_ic, 0, 1)
    
    assert mock_plot.call_count == 5

def test_plot_ens_sample_init(mock_config):
    plot = PlotEnsSample(mock_config)
    
    assert plot.sample_idx == 0
    assert plot.lead_time_to_eval == [1, 2, 3, 4, 5]

@patch('anemoi.training.diagnostics.callbacks.forecast_ensemble.plot_predicted_ensemble')
def test_plot_ens_sample_generate_plot_fn(mock_plot, mock_config):
    plot = PlotEnsSample(mock_config)
    
    plot_parameters_dict = {"temperature": (0, True), "pressure": (1, True)}
    latlons = np.random.rand(100, 2)
    x = np.random.rand(10, 100, 3)
    y_true = np.random.rand(10, 100, 3)
    y_pred = np.random.rand(10, 100, 3)
    
    plot._generate_plot_fn(plot_parameters_dict, 4, latlons, [1, 2, 3], "viridis", x, y_true, y_pred, False)
    
    mock_plot.assert_called_once()

def test_plot_ens_power_spectrum_init(mock_config):
    plot = PlotEnsPowerSpectrum(mock_config, val_dset_len=100)
    
    assert plot.ens_indexes_to_plot == [0, 1]

@patch('anemoi.training.diagnostics.callbacks.forecast_ensemble.plot_power_spectrum')
def test_plot_ens_power_spectrum_plot_spectrum(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = PlotEnsPowerSpectrum(mock_config, val_dset_len=100)
    plot.latlons = np.random.rand(100, 2)
    
    outputs = {
        "data": np.random.rand(10, 5, 100, 3),
        "y_pred_postprocessed": torch.rand(5, 10, 100, 3)
    }
    batch = torch.rand(10, 5, 100, 3)
    
    plot._plot_spectrum(mock_trainer, mock_pl_module, outputs, batch, 0, 1)
    
    assert mock_plot.call_count == 5

def test_setup_rollout_eval_callbacks(mock_config):
    callbacks = setup_rollout_eval_callbacks(mock_config, val_dset_len=100)
    
    assert isinstance(callbacks, RolloutEvalEns)
    assert len(callbacks.callbacks_validation_batch_end) == 7
    assert len(callbacks.callbacks_validation_epoch_end) == 7

def test_get_callbacks(mock_config):
    monitored_metrics = ["val_loss"]
    callbacks = get_callbacks(mock_config, monitored_metrics, val_dset_len=100)
    
    callback_types = [type(cb) for cb in callbacks]
    assert RolloutEvalEns in callback_types

# Add more tests as needed for other callbacks and edge cases