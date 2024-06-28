#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import json
import logging
import os
import re

import hydra
from anemoi.utils.config import config_path
from omegaconf import OmegaConf

from . import Command

LOGGER = logging.getLogger(__name__)

# https://hydra.cc/docs/advanced/override_grammar/basic/

override_regex = re.compile(
    r"""
        ^
        (
            (~|\+|\+\+)?       # optional prefix
            (\w+)([/@:\.]\w+)* # key
            =                  # assignment
            (.*)               # value
        )
        |                      # or
        (~                     # ~ prefix
            (\w+)([/@:\.]\w+)  # key
        )
        $
    """,
    re.VERBOSE,
)


class Train(Command):

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "config",
            nargs="*",
            type=str,
            help="A list yaml files to load or a list of overrides to apply",
        )

    def run(self, args):

        configs = []
        overrides = []

        for config in args.config:
            if override_regex.match(config):
                overrides.append(config)
            elif config.endswith(".yaml") or config.endswith(".yml"):
                configs.append(config)
            else:
                raise ValueError(f"Invalid config '{config}'. It must be a yaml file or an override")

        hydra.initialize(config_path="../config", version_base=None)

        cfg = hydra.compose(config_name="config")  # , overrides=overrides)
        print(cfg)

        # Add user config
        user_config = config_path("train.yaml")

        if os.path.exists(user_config):
            LOGGER.info(f"Loading config {user_config}")
            cfg = OmegaConf.merge(cfg, OmegaConf.load(user_config))

        # Add extra config files specified in the command line

        for config in configs:
            LOGGER.info(f"Loading config {config}")
            cfg = OmegaConf.merge(cfg, OmegaConf.load(config))

        # We need to reapply the overrides
        # This does not support overrides with a prefix
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

        print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

        # AIFSTrainer(cfg).train()


command = Train
