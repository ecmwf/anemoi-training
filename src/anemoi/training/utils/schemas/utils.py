# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from pydantic import BaseModel
from pydantic import Field


class HydraInstantiable(BaseModel):
    target_: str = Field(..., alias="_target_")
    convert_: str = Field("all", alias="_convert_")
