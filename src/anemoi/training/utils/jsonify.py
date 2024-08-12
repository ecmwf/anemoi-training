# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import BaseIndex
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.data_indices.tensor import BaseTensorIndex


def map_config_to_primitives(config: OmegaConf) -> dict:
    """Ensure that the metadata information is JSON-serializable.

    Parameters
    ----------
    config : OmegaConf
        config object to be mapped to primitives.

    Returns
    -------
    dict
        JSON serializable dictionary.

    Raises
    ------
    TypeError
        Cannot map object to primitives.

    """
    if config is None or isinstance(config, (int, float, str, bool)):
        return config

    if isinstance(config, Path):
        config = str(config)
    elif isinstance(config, datetime.date):
        config = config.isoformat()
    elif isinstance(config, (list, tuple)):
        config = [map_config_to_primitives(v) for v in config]
    elif isinstance(config, dict):
        config = {k: map_config_to_primitives(v) for k, v in config.items()}
    elif isinstance(config, DictConfig):
        config = map_config_to_primitives(dict(config))
    elif isinstance(config, ListConfig):
        config = map_config_to_primitives(list(config))
    elif isinstance(config, torch.Tensor):
        config = map_config_to_primitives(config.tolist())
    elif isinstance(config, (IndexCollection, BaseTensorIndex, BaseIndex)):
        config = map_config_to_primitives(config.todict())
    else:
        msg = f"Cannot serialize object of type {type(config)}"
        raise TypeError(msg)

    return config
