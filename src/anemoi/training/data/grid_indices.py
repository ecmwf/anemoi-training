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
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)

ArrayIndex = Union[slice, int, Sequence[int]]


class BaseGridIndices(ABC):
    """Base class for custom grid indices."""

    def __init__(self, nodes_name: str) -> None:
        self.nodes_name = nodes_name

    def setup(self, graph: HeteroData) -> None:
        self.grid_size = self.compute_grid_size(graph)

    def split_seq_in_shards(self, reader_group_rank: int, reader_group_size: int) -> tuple[int, int]:
        """Get the indices to split a sequence into equal size shards."""
        grid_shard_size = self.grid_size // reader_group_size
        grid_start = reader_group_rank * grid_shard_size
        if reader_group_rank == reader_group_size - 1:
            grid_end = self.grid_size
        else:
            grid_end = (reader_group_rank + 1) * grid_shard_size

        return slice(grid_start, grid_end)

    @abstractmethod
    def compute_grid_size(self, graph: HeteroData) -> int: ...

    @abstractmethod
    def get_shard_indices(self, reader_group_rank: int, reader_group_size: int) -> ArrayIndex: ...


class FullGrid(BaseGridIndices):
    """The full grid is loaded."""

    def compute_grid_size(self, graph: HeteroData) -> int:
        return graph[self.nodes_name].num_nodes

    def get_shard_indices(self, reader_group_rank: int, reader_group_size: int) -> ArrayIndex:
        return self.split_seq_in_shards(reader_group_rank, reader_group_size)


class MaskedGrid(BaseGridIndices):
    """Grid is masked based on a node attribute."""

    def __init__(self, nodes_name: str, node_attribute_name: str):
        super().__init__(nodes_name)
        self.node_attribute_name = node_attribute_name

    def setup(self, graph: HeteroData) -> None:
        LOGGER.info(
            "The graph attribute %s of the %s nodes will be used to masking the spatial dimension.",
            self.node_attribute_name,
            self.nodes_name,
        )
        self.spatial_indices = graph[self.nodes_name][self.node_attribute_name].squeeze().tolist()
        super().setup(graph)

    def compute_grid_size(self, _graph: HeteroData) -> int:
        return len(self.spatial_indices)

    def get_shard_indices(self, reader_group_rank: int, reader_group_size: int) -> ArrayIndex:
        sequence_indices = self.split_seq_in_shards(reader_group_rank, reader_group_size)
        return self.spatial_indices[sequence_indices]
