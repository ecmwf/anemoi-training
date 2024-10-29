# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import os
import sys

from anemoi.training.commands.train import TrainBase

LOGGER = logging.getLogger(__name__)


class Profiler(TrainBase):
    """Commands to profile Anemoi models."""

    accept_unknown_args = True

    def run(self, args: list[str], unknown_args: list[str] | None = None) -> None:
        # This will be picked up by the logger
        os.environ["ANEMOI_TRAINING_CMD"] = f"{sys.argv[0]} {args.command}"
        # Merge the known subcommands with a non-whitespace character for hydra
        new_sysargv = self._merge_sysargv(args)

        # Add the unknown arguments (belonging to hydra) to sys.argv
        if unknown_args is not None:
            sys.argv = [new_sysargv, *unknown_args]
        else:
            sys.argv = [new_sysargv]

        # Import and run the profiler command
        LOGGER.info("Running anemoi profiling command with overrides: %s", sys.argv[1:])
        main()


def main() -> None:
    # Use the environment variable to check if main is being called from the subcommand, not from the ddp entrypoint
    if not os.environ.get("ANEMOI_TRAINING_CMD"):
        error = "This entrypoint should not be called directly. Use `anemoi-training profiler` instead."
        raise RuntimeError(error)

    from anemoi.training.train.profiler import main as anemoi_profiler

    anemoi_profiler()


command = Profiler
