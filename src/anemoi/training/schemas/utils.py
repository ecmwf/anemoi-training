# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from typing import Any


class ValidationError(Exception):
    pass


def allowed_values(v: Any, values: list[Any]) -> Any:
    if v not in values:
        msg = {f"Value {v} not in {values}"}
        raise ValidationError(msg)
    return v
