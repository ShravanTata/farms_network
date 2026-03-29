""" Molkov oscillator (Molkov et al. 2015) """

from farms_network.core.node import Node
from farms_network.core.edge import Edge
from farms_network.models import Models
from farms_network.models.molkov_oscillator_cy import MolkovOscillatorNodeCy
from farms_network.models.molkov_oscillator_cy import MolkovOscillatorEdgeCy


class MolkovOscillatorNode(Node):

    CY_NODE_CLASS = MolkovOscillatorNodeCy

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model=Models.MOLKOV_OSCILLATOR, **kwargs)


class MolkovOscillatorEdge(Edge):

    CY_EDGE_CLASS = MolkovOscillatorEdgeCy

    def __init__(self, source, target, edge_type, model=Models.MOLKOV_OSCILLATOR, **kwargs):
        super().__init__(
            source=source, target=target, edge_type=edge_type,
            model=Models.MOLKOV_OSCILLATOR, **kwargs
        )
