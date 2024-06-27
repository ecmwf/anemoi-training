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
import sys

import hydra
from anemoi.utils.config import load_raw_config
from omegaconf import OmegaConf

from . import Command


class Train(Command):

    def add_arguments(self, command_parser):
        print("aaa")
        command_parser.add_argument("--main", action="store_true", help="Run the main function")
        command_parser.add_argument("--config", nargs="*", type=str, help="A list of extra config files to load")

    def run(self, args):
        # Just a proof of concept
        if args.main:

            @hydra.main(config_path="../config", config_name="config")
            def hydra_main(cfg):
                print(dir(cfg))
                print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

            del sys.argv[1]  # train
            del sys.argv[1]  # --main
            hydra_main()
            exit(0)

        hydra.initialize(config_path="../config")

        cfg = hydra.compose(config_name="config")

        # Add project config
        # cfg = OmegaConf.merge(cfg, OmegaConf.create(...))

        # Add experiment config
        # cfg = OmegaConf.merge(cfg, OmegaConf.create(...))

        # Add user config
        cfg = OmegaConf.merge(cfg, OmegaConf.create(load_raw_config("training.yaml", default={})))

        # Add extra config files specified in the command line
        if args.config:
            for config in args.config:
                print(f"Loading config {config}")
                cfg = OmegaConf.merge(cfg, OmegaConf.load(config))

        print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))


command = Train
