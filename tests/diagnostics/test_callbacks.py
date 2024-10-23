# ruff: noqa: ANN001, ANN201
# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import omegaconf
import pytest
import yaml

from anemoi.training.diagnostics.callbacks import get_callbacks

default_config = """
diagnostics:
  callbacks: []

  plot:
    enabled: False
    callbacks: []

  debug:
    # this will detect and trace back NaNs / Infs etc. but will slow down training
    anomaly_detection: False

  profiler: False

<<<<<<< HEAD
  enable_checkpointing: False
  checkpoint:
=======
  checkpoint:
    enabled: False
>>>>>>> 3a0df9d (rebase)

  log: {}
"""


def test_no_extra_callbacks_set():
    # No extra callbacks set
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    callbacks = get_callbacks(config)
    assert len(callbacks) == 1  # ParentUUIDCallback


<<<<<<< HEAD
=======
def test_deprecation_warning():
    # Test deprecation warning
    with pytest.warns(DeprecationWarning):
        config = omegaconf.OmegaConf.create(default_config)
        config.diagnostics.update({"eval": {"enabled": True, "rollout": 1, "frequency": 1}})
        callbacks = get_callbacks(config)
        assert len(callbacks) == 2


>>>>>>> 3a0df9d (rebase)
def test_add_config_enabled_callback():
    # Add logging callback
    config = omegaconf.OmegaConf.create(default_config)
    config.diagnostics.callbacks.append({"log": {"mlflow": {"enabled": True}}})
    callbacks = get_callbacks(config)
    assert len(callbacks) == 2


def test_add_callback():
    config = omegaconf.OmegaConf.create(default_config)
<<<<<<< HEAD
    config.diagnostics.callbacks.append(
        {"_target_": "anemoi.training.diagnostics.callbacks.provenance.ParentUUIDCallback"},
    )
=======
    config.diagnostics.callbacks.append({"_target_": "anemoi.training.diagnostics.callbacks.id.ParentUUIDCallback"})
>>>>>>> 3a0df9d (rebase)
    callbacks = get_callbacks(config)
    assert len(callbacks) == 2


def test_add_plotting_callback(monkeypatch):
    # Add plotting callback
    import anemoi.training.diagnostics.callbacks.plot as plot

    class PlotLoss:
        def __init__(self, config: omegaconf.DictConfig):
            pass

    monkeypatch.setattr(plot, "PlotLoss", PlotLoss)

    config = omegaconf.OmegaConf.create(default_config)
    config.diagnostics.plot.enabled = True
    config.diagnostics.plot.callbacks = [{"_target_": "anemoi.training.diagnostics.callbacks.plot.PlotLoss"}]
    callbacks = get_callbacks(config)
    assert len(callbacks) == 2
