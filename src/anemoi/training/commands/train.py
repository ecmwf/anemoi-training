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
from hydra.errors import ConfigCompositionException
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


def apply_delete_override(cfg, dotkey, value, parent, key, value_given):

    any_value = object()

    if value_given:
        value_given = value
    else:
        value_given = any_value

    value = OmegaConf.select(cfg, dotkey, throw_on_missing=False)
    if value_given is not any_value and value != value_given:
        raise ConfigCompositionException(
            f"Key '{dotkey}' with value '{value}' does not match the value '{value_given}' in the override"
        )

    try:
        # Allow 'del'
        OmegaConf.set_struct(cfg, False)
        if key is None:
            # Top level key
            del cfg[parent]
        else:
            subtree = OmegaConf.select(cfg, parent)
            del subtree[key]
    finally:
        OmegaConf.set_struct(cfg, True)


def apply_add_override_force(cfg, dotkey, value, parent, key):
    OmegaConf.update(cfg, dotkey, value, merge=True, force_add=True)


def apply_add_override(cfg, dotkey, value, parent, key):
    current = OmegaConf.select(cfg, dotkey, throw_on_missing=False)
    if current is not None:
        raise ConfigCompositionException(f"Cannot add key '{dotkey}' because it already exists, use '++' to force add")

    OmegaConf.update(cfg, dotkey, value, merge=True, force_add=True)


def apply_assign_override(cfg, dotkey, value, parent, key):
    OmegaConf.update(cfg, dotkey, value, merge=True)


def parse_override(override, n):
    dotkey = override[n:]
    parsed = OmegaConf.from_dotlist([dotkey])
    dotkey = dotkey.split("=")[0]
    value = OmegaConf.select(parsed, dotkey)

    if "." in dotkey:
        parent, key = dotkey.rsplit(".", 1)
        return dotkey, value, parent, key
    else:
        return dotkey, value, dotkey, None


def apply_override(cfg, override):
    if override.startswith("~"):
        return apply_delete_override(cfg, *parse_override(override, 1), value_given="=" in override)

    if override.startswith("++"):
        return apply_add_override_force(cfg, *parse_override(override, 2))

    if override.startswith("+"):
        return apply_add_override(cfg, *parse_override(override, 1))

    return apply_assign_override(cfg, *parse_override(override, 0))


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

        cfg = hydra.compose(config_name="config")

        # Add user config
        user_config = config_path("train.yaml")

        if os.path.exists(user_config):
            LOGGER.info(f"Loading config {user_config}")
            cfg = OmegaConf.merge(cfg, OmegaConf.load(user_config, resolve=True))

        # Add extra config files specified in the command line

        for config in configs:
            LOGGER.info(f"Loading config {config}")
            cfg = OmegaConf.merge(cfg, OmegaConf.load(config))

        # We need to reapply the overrides
        # OmegaConf do not implement the prefix logic, this is done by hydra
        for override in overrides:
            LOGGER.info(f"Applying override {override}")
            apply_override(cfg, override)

        # Resolve the config
        OmegaConf.resolve(cfg)

        print(json.dumps(OmegaConf.to_container(cfg), indent=4))

        # AIFSTrainer(cfg).train()


command = Train
