# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from _pytest.fixtures import SubRequest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
import sys

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule


@pytest.fixture
def config(request: SubRequest) -> DictConfig:
    overrides = request.param
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        return compose(config_name="debug", overrides=overrides)


@pytest.fixture
def datamodule() -> AnemoiDatasetsDataModule:
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return AnemoiDatasetsDataModule(cfg)

# enable_stop_on_exceptions if the debugger is running during a test
def is_debugging():
    return 'debugpy' in sys.modules

if is_debugging():
  @pytest.hookimpl(tryfirst=True)
  def pytest_exception_interact(call):
    raise call.excinfo.value
    
  @pytest.hookimpl(tryfirst=True)
  def pytest_internalerror(excinfo):
    raise excinfo.value
