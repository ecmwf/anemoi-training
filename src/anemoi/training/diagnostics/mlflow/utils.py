# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from collections.abc import Iterable
from typing import Any

import requests


def health_check(tracking_uri: str) -> None:
    """Query the health endpoint of an MLflow server.

    If the server is not reachable, raise an error and remind the user that authentication may be required.

    Raises
    ------
    ConnectionError
        If the server is not reachable.

    """
    token = os.getenv("MLFLOW_TRACKING_TOKEN")

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{tracking_uri}/health", headers=headers, timeout=60)

    if response.text == "OK":
        return

    error_msg = f"Could not connect to MLflow server at {tracking_uri}. "
    if not token:
        error_msg += "The server may require authentication, did you forget to turn it on?"
    raise ConnectionError(error_msg)


def expand_iterables(params: dict[str, Any]) -> dict[str, Any]:
    """Expand any iterable values in the dictionary to a dictionary of the form {key[i]: value_i}.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters to be expanded.

    Returns
    -------
    dict[str, Any]
        Dictionary with all iterable values expanded.

    Examples
    --------
        >>> expand_iterables({'a': ['a', 'b', 'c']})
        {'a[0]': 'a', 'a[1]': 'b', 'a[2]': 'c'}
        >>> expand_iterables({'a': {'b': ['a', 'b', 'c']}})
        {'a': {'b[0]': 'a', 'b[1]': 'b', 'b[2]': 'c'}}
    """
    expanded_params = {}
    for key, value in params.items():
        if isinstance(value, Iterable) and not isinstance(value, str):
            expanded_params[key] = {
                f"{key}[{i}]": expand_iterables(v) if isinstance(v, dict) else v for i, v in enumerate(value)
            }
            expanded_params[key] = value
            expanded_params[f"{key}_length"] = len(value)
        else:
            expanded_params[key] = expand_iterables(value) if isinstance(value, dict) else value
    return expanded_params
