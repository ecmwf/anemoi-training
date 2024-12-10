# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from hydra import compose
from hydra import initialize
from hydra.core.config_store import ConfigStore

from anemoi.training.utils.data_classes import TrainingConfig

cs = ConfigStore.instance()
cs.store(name="training", node=TrainingConfig)


@pytest.fixture
def training_config() -> TrainingConfig:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config/training"):
        return compose(config_name="default")


def test_config_installed(training_config: TrainingConfig) -> None:

    assert training_config.run_id is None
