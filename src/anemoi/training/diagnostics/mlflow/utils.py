# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import functools
import os
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


def expand_iterables(
    params: dict[str, Any],
    *,
    size_threshold: int | None = None,
    recursive: bool = True,
) -> dict[str, Any]:
    """Expand any iterable values in the dictionary to a dictionary of the form {key_i: value_i}.

    If `size_threshold` is not None, expand the iterable only if the length of str(value) is
    greater than `size_threshold`.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters to be expanded.
    size_threshold : int | None, optional
        Threshold of str(value) to expand iterable at.
        Default is None.
    recursive : bool, optional
        Expand nested dictionaries.
        Default is True.

    Returns
    -------
    dict[str, Any]
        Dictionary with all iterable values expanded.

    Examples
    --------
        >>> expand_iterables({'a': ['a', 'b', 'c']})
        {'a_0': 'a', 'a_1': 'b', 'a_2': 'c'}
        >>> expand_iterables({'a': {'b': ['a', 'b', 'c']}})
        {'a': {'b_0': 'a', 'b_1': 'b', 'b_2': 'c'}}
    """
    nested_func = functools.partial(expand_iterables, size_threshold=size_threshold, recursive=recursive)

    expanded_params = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)) and (size_threshold is None or len(str(value)) > size_threshold):
            for i, v in enumerate(value):
                expanded_params[f"{key}_{i}"] = nested_func(v) if isinstance(v, dict) and recursive else v

            expanded_params[f"{key}_all"] = value
            expanded_params[f"{key}_length"] = len(value)
        else:
            expanded_params[key] = nested_func(value) if isinstance(value, dict) and recursive else value
    return expanded_params
