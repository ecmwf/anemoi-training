#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from anemoi.training.diagnostics.mlflow.auth import TokenAuth

from . import Command


class MlFlow(Command):
    """Commands to interact with MLflow."""

    def add_arguments(self, command_parser):
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        login = subparsers.add_parser(
            "login",
            help="Log in and acquire a token from keycloak.",
        )
        login.add_argument(
            "--url",
            help="The URL of the authentication server",
        )
        login.add_argument(
            "--force-credentials",
            "-f",
            action="store_true",
            help="Force a credential login even if a token is available.",
        )

    def run(self, args):
        if args.subcommand == "login":
            TokenAuth(url=args.url).login(force_credentials=args.force_credentials)


command = MlFlow
