import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from anemoi.training.diagnostics.callbacks import (
    BasePlotCallback,
    GraphTrainableFeaturesPlot,
    BaseLossBarPlot,
    WeightGradOutputLoggerCallback,
    ParentUUIDCallback,
    BaseLossMapPlot,
    AnemoiCheckpoint,
    MemCleanUpCallback,
    VideoPlotCallback,
)

@pytest.fixture
def mock_config():
    return OmegaConf.create({
        "hardware": {"paths": {"plots": "/tmp/plots"}},
        "diagnostics": {
            "plot": {
                "frequency": 10,
                "asynchronous": False,
                "parameter_groups": {"group1": ["param1", "param2"]},
            },
            "log": {
                "wandb": {"enabled": False},
                "mlflow": {"enabled": False},
            },
        },
        "graph": {
            "data": "data_graph",
            "hidden": "hidden_graph",
        },
        "dataloader": {
            "batch_size": {"test": 1},
        },
    })

@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=pl.Trainer)
    trainer.logger = MagicMock()
    trainer.current_epoch = 0
    trainer.global_step = 0
    return trainer

@pytest.fixture
def mock_pl_module():
    pl_module = MagicMock(spec=pl.LightningModule)
    pl_module.local_rank = 0
    pl_module.data_indices = MagicMock()
    pl_module.data_indices.internal_model.output.name_to_index = {"param1": 0, "param2": 1}
    pl_module.multi_step = 5
    pl_module.rollout = 10
    return pl_module


def test_base_loss_bar_plot(mock_config, mock_trainer, mock_pl_module):
    callback = BaseLossBarPlot(mock_config)
    
    outputs = [torch.randn(1, 10, 5, 2), torch.randn(1, 10, 5, 2)]
    batch = torch.randn(1, 15, 5, 2)
    
    with patch.object(callback, 'plot') as mock_plot:
        callback.on_validation_batch_end(mock_trainer, mock_pl_module, outputs, batch, 0)
        mock_plot.assert_called_once()






