""" ReLU """

from farms_network.core.node import Node
from farms_network.models import Models
from farms_network.core.options import ReLUNodeOptions
from farms_network.models.relu_cy import ReLUNodeCy


class ReLUNode(Node):

    CY_NODE_CLASS = ReLUNodeCy

    def __init__(self, name: str):
        super().__init__(name=name, model=Models.RELU)
