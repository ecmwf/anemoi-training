# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod

from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class BaseSpatialIndices(ABC):
    """Base class for custom spatial indices."""

    @abstractmethod
    def get_indices(self, graph: HeteroData) -> list[int] | None: ...


class NoSpatialIndices(BaseSpatialIndices):
    """No spatial mask."""

    def get_indices(self, graph: HeteroData) -> None:
        return None


class GraphNodeAttribute(BaseSpatialIndices):
    """Get graph node attribute."""

    def __init__(self, nodes_name: str, node_attribute_name: str):
        self.nodes_name = nodes_name
        self.node_attribute_name = node_attribute_name

    def get_indices(self, graph: HeteroData) -> list[int]:
        LOGGER.info(
            "The graph attribute %s of the %s nodes will be used to masking the spatial dimension.",
            self.node_attribute_name,
            self.nodes_name,
        )
        return graph[self.nodes_name][self.node_attribute_name].squeeze().tolist()
