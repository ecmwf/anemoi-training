# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import torch
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from scipy.spatial import SphericalVoronoi
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphNodeAttribute:
    """Base class to load and optionally change the weight attribute of nodes in the graph.

    Attributes
    ----------
    target: str
        name of target nodes, key in HeteroData graph object
    node_attribute: str
        name of node weight attribute, key in HeteroData graph object

    Methods
    -------
    weights(self, graph_data)
        Load node weight attribute. Compute area weights if they can not be found in graph
        object.
    """

    def __init__(self, target_nodes: str, node_attribute: str):
        """Initialize graph node attribute with target nodes and node attribute.

        Parameters
        ----------
        target_nodes: str
            name of nodes, key in HeteroData graph object
        node_attribute: str
            name of node weight attribute, key in HeteroData graph object
        """
        self.target = target_nodes
        self.node_attribute = node_attribute

    def area_weights(self, graph_data: HeteroData) -> np.ndarray:
        """Nodes weighted by the size of the geographical area they represent.

        Parameters
        ----------
        graph_data: HeteroData
            graph object

        Returns
        -------
        np.ndarray
            area weights of the target nodes
        """
        lats, lons = graph_data[self.target].x[:, 0], graph_data[self.target].x[:, 1]
        points = latlon_rad_to_cartesian((np.asarray(lats), np.asarray(lons)))
        sv = SphericalVoronoi(points, radius=1.0, center=[0.0, 0.0, 0.0])
        area_weights = sv.calculate_areas()

        return area_weights / np.max(area_weights)

    def weights(self, graph_data: HeteroData) -> torch.Tensor:
        """Returns weight of type self.node_attribute for nodes self.target.

        Attempts to load from graph_data and calculates area weights for the target
        nodes if they do not exist.

        Parameters
        ----------
        graph_data: HeteroData
            graph object

        Returns
        -------
        torch.Tensor
            weight of target nodes
        """
        if self.node_attribute in graph_data[self.target]:
            attr_weight = graph_data[self.target][self.node_attribute].squeeze()

            LOGGER.info("Loading node attribute %s from the graph", self.node_attribute)
        else:
            attr_weight = torch.from_numpy(self.area_weights(graph_data))

            LOGGER.info(
                "Node attribute %s not found in graph. Default area weighting will be used",
                self.node_attribute,
            )

        return attr_weight


class ReweightedGraphNodeAttribute(GraphNodeAttribute):
    """Method to reweight a subset of the target nodes defined by scaled_attribute.

    Subset nodes will be scaled such that their weight sum equals weight_frac_of_total of the sum
    over all nodes.
    """

    def __init__(self, target_nodes: str, node_attribute: str, scaled_attribute: str, weight_frac_of_total: float):
        """Initialize reweighted graph node attribute.

        Parameters
        ----------
        target_nodes: str
            name of nodes, key in HeteroData graph object
        node_attribute: str
            name of node weight attribute, key in HeteroData graph object
        scaled_attribute: str
            name of node attribute defining the subset of nodes to be scaled, key in HeteroData graph object
        weight_frac_of_total: float
            sum of weight of subset nodes as a fraction of sum of weight of all nodes after rescaling

        """
        super().__init__(target_nodes=target_nodes, node_attribute=node_attribute)
        self.scaled_attribute = scaled_attribute
        self.fraction = weight_frac_of_total

    def weights(self, graph_data: HeteroData) -> torch.Tensor:
        attr_weight = super().weights(graph_data)

        mask = graph_data[self.target][self.scaled_attribute].squeeze().bool()
        unmasked_sum = torch.sum(attr_weight[~mask])
        weight_per_masked_node = self.fraction / (1 - self.fraction) * unmasked_sum / sum(mask)
        attr_weight[mask] = weight_per_masked_node

        LOGGER.info(
            "Weight of nodes in %s rescaled such that their sum equals %.3f of the sum over all nodes",
            self.scaled_attribute,
            self.fraction,
        )

        return attr_weight
