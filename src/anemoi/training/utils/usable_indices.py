# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import numpy as np


def get_usable_indices(
    missing_indices: set[int] | None,
    series_length: int,
    relative_indices: np.ndarray,
) -> np.ndarray:
    """Get the usable indices of a series whit missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Dataset to be used.
    series_length : int
        Length of the series.
    relative_indices:
        Array of relative indices requested at each index i.

    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    usable_indices = np.arange(series_length)  # set of all indices

    if missing_indices is None:
        missing_indices = set()

    missing_indices |= {series_length} #filter final index

    # Missing indices
    for i in missing_indices:
        rel_missing = i - relative_indices #indices which have their relative indices match the missing.
        usable_indices = usable_indices[np.all(usable_indices != rel_missing[:,np.newaxis], axis = 0)]

    return usable_indices
