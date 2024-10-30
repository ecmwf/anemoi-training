import pytest
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
from anemoi.training.diagnostics.callbacks.reconstruction import (
    ReconstructionLossBarPlot,
    ReconstructionLossMapPlot,
    PlotReconstructedSample,
    SpectralAnalysisPlot,
    ReconstructEval,
    setup_reconstruction_eval_callbacks,
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
                "parameters_spectrum": ["temperature", "pressure"],
                "sample_idx": 0,
                "per_sample": 1,
                "loss_map": True,
                "loss_bar": True,
                "plot_spectral_loss": True,
                "plot_reconstructed_sample": True
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
    module.data_indices.data.output.full = slice(None)
    module.latlons_data = torch.rand(100, 2)
    module.latlons_hidden = torch.rand(50, 2)
    module.loss = MagicMock()
    module.loss.name = "mse"
    module.loss.reconstruction_loss = MagicMock()
    module.loss.reconstruction_loss.name = "recon_loss"
    module.loss.divergence_loss = MagicMock()
    module.loss.divergence_loss.name = "div_loss"
    module.global_rank = 0
    module.local_rank = 0
    module.model = MagicMock()
    module.model.normalizer = MagicMock()
    return module

@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=pl.Trainer)
    trainer.logger = MagicMock()
    trainer.current_epoch = 0
    trainer.global_step = 0
    return trainer

def test_reconstruction_loss_bar_plot_init(mock_config):
    plot = ReconstructionLossBarPlot(mock_config, val_dset_len=100)
    
    assert plot.counter == 0
    assert isinstance(plot.loss_map_accum, dict)

def test_reconstruction_loss_bar_plot_accumulate(mock_config, mock_pl_module, mock_trainer):
    plot = ReconstructionLossBarPlot(mock_config, val_dset_len=100)
    
    outputs = {
        1: {
            "x_target": torch.rand(10, 100, 3),
            "x_rec": torch.rand(10, 100, 3),
            "z_mu": torch.rand(10, 50),
            "z_logvar": torch.rand(10, 50)
        }
    }
    batch = torch.rand(10, 5, 100, 3)
    
    plot.accumulate(mock_trainer, mock_pl_module, outputs, batch)
    
    assert "reconstruction" in plot.loss_map_accum
    assert plot.loss_map_accum["reconstruction"] is not None

@patch('anemoi.training.diagnostics.callbacks.reconstruction.plot_loss')
def test_reconstruction_loss_bar_plot_plot(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = ReconstructionLossBarPlot(mock_config, val_dset_len=100)
    plot.loss_map_accum["reconstruction"] = torch.rand(3)
    plot.counter = 1
    
    plot._plot(mock_trainer, mock_pl_module, None, None, 0, 1)
    
    mock_plot.assert_called_once()

def test_reconstruction_loss_map_plot_init(mock_config):
    plot = ReconstructionLossMapPlot(mock_config, val_dset_len=100)
    
    assert isinstance(plot.loss_map_accum, dict)

def test_reconstruction_loss_map_plot_accumulate(mock_config, mock_pl_module):
    plot = ReconstructionLossMapPlot(mock_config, val_dset_len=100)
    
    outputs = {
        1: {
            "x_target": torch.rand(10, 100, 3),
            "x_rec": torch.rand(10, 100, 3),
            "z_mu": torch.rand(10, 50),
            "z_logvar": torch.rand(10, 50)
        }
    }
    
    plot.accumulate(mock_trainer, mock_pl_module, outputs)
    
    assert "reconstruction" in plot.loss_map_accum
    assert "divergence" in plot.loss_map_accum

@patch('anemoi.training.diagnostics.callbacks.reconstruction.plot_loss_map')
def test_reconstruction_loss_map_plot_plot(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = ReconstructionLossMapPlot(mock_config, val_dset_len=100)
    plot.loss_map_accum["reconstruction"] = torch.rand(100, 3)
    plot.loss_map_accum["divergence"] = torch.rand(50, 1)
    plot.counter = 1
    
    plot._plot(mock_trainer, mock_pl_module)
    
    assert mock_plot.call_count == 2

def test_plot_reconstructed_sample_init(mock_config):
    plot = PlotReconstructedSample(mock_config, val_dset_len=100)
    
    assert plot.sample_idx == 0

@patch('anemoi.training.diagnostics.callbacks.reconstruction.plot_reconstructed_multilevel_sample')
def test_plot_reconstructed_sample_plot(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = PlotReconstructedSample(mock_config, val_dset_len=100)
    
    outputs = {
        1: {
            "x_inp": torch.rand(10, 1, 100, 3),
            "x_rec": torch.rand(10, 100, 3)
        }
    }
    batch = torch.rand(10, 5, 100, 3)
    
    plot._plot(mock_trainer, mock_pl_module, outputs, batch, 0, 1)
    
    mock_plot.assert_called_once()

def test_spectral_analysis_plot_init(mock_config):
    plot = SpectralAnalysisPlot(mock_config, val_dset_len=100)
    
    assert plot.sample_idx == 0

@patch('anemoi.training.diagnostics.callbacks.reconstruction.plot_power_spectrum')
def test_spectral_analysis_plot_plot_spectrum(mock_plot, mock_config, mock_pl_module, mock_trainer):
    plot = SpectralAnalysisPlot(mock_config, val_dset_len=100)
    
    outputs = {
        "x_inp_postprocessed": torch.rand(10, 5, 100, 3),
        "x_target_postprocessed": torch.rand(10, 5, 100, 3),
        "x_rec_postprocessed": torch.rand(10, 5, 100, 3)
    }
    batch = torch.rand(10, 5, 100, 3)
    
    plot._plot_spectrum(mock_trainer, mock_pl_module, outputs, batch, 0, 1)
    
    assert mock_plot.call_count == 5

def test_reconstruct_eval_init(mock_config):
    callbacks = [MagicMock(), MagicMock()]
    reconstruct_eval = ReconstructEval(mock_config, val_dset_len=100, callbacks=callbacks)
    
    assert reconstruct_eval.frequency == 1
    assert reconstruct_eval.eval_frequency == 1
    assert len(reconstruct_eval.callbacks_validation_batch_end) == 2
    assert len(reconstruct_eval.callbacks_validation_epoch_end) == 2

def test_reconstruct_eval_on_validation_batch_end(mock_config, mock_pl_module, mock_trainer):
    callbacks = [MagicMock(), MagicMock()]
    reconstruct_eval = ReconstructEval(mock_config, val_dset_len=100, callbacks=callbacks)
    
    batch = torch.rand(10, 5, 100, 3)
    outputs = {"loss": torch.tensor(0.1), "metrics": {"metric1": 0.5}}
    
    with patch.object(reconstruct_eval, '_eval', return_value=outputs):
        reconstruct_eval.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, batch, 0)
    
    for callback in callbacks:
        callback.on_validation_batch_end.assert_called_once()

def test_setup_reconstruction_eval_callbacks(mock_config):
    callbacks = setup_reconstruction_eval_callbacks(mock_config, val_dset_len=100)
    
    assert isinstance(callbacks, ReconstructEval)
    assert len(callbacks.callbacks_validation_batch_end) == 4
    assert len(callbacks.callbacks_validation_epoch_end) == 4

def test_get_callbacks(mock_config):
    monitored_metrics = ["val_loss"]
    callbacks = get_callbacks(mock_config, monitored_metrics, val_dset_len=100)
    
    callback_types = [type(cb) for cb in callbacks]
    assert ReconstructEval in callback_types

# Add more tests as needed for edge cases and other scenarios
