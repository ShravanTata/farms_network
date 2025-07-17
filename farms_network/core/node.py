""" Node """


from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from farms_network.core.node_cy import NodeCy
from farms_network.core.options import NodeOptions


class Node(ABC):

    CY_NODE_CLASS: Type[NodeCy] = None

    def __init__(self, name: str, model: str, **parameters):
        self.name: str = name        # Unique name of the node
        self.model: str = model      # Type of the model (e.g., "empty")
        self._node_cy = self._create_cy_node(**parameters)

    def _create_cy_node(self, **kwargs) -> NodeCy:
        if self.CY_NODE_CLASS is None:
            raise NotImplementedError("Must define CY_NODE_CLASS")
        return self.CY_NODE_CLASS(**kwargs)

    # General node properties
    @property
    def nstates(self):
        return self._node_cy.nstates

    @property
    def nparams(self):
        return self._node_cy.nparams

    @property
    def ninputs(self):
        return self._node_cy.ninputs

    def print_parameters(self):
        return self._node_cy.parameters

    @classmethod
    def from_options(cls, node_options: NodeOptions):
        """ From node options """
        name: str = node_options.name
        parameters = node_options.parameters
        return cls(name, **parameters)

    def to_options(self):
        """ To node options """
        name: str = node_options.name
        parameters = node_options.parameters
        return cls(name, **parameters)
