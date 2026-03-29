""" Network """

from typing import List, Optional, Self

import numpy as np
from farms_core import pylog
from farms_network.numeric import integrators

from ..models.factory import EdgeFactory, NodeFactory
from ..noise.ornstein_uhlenbeck import OrnsteinUhlenbeck
from .data import NetworkData, NetworkLog
from .edge import Edge
from .network_cy import NetworkCy
from .node import Node
from .options import (EdgeOptions, IntegrationOptions, NetworkOptions,
                      NodeOptions)


class Network:
    """ Network class using composition with NetworkCy """

    def __init__(self, network_options: NetworkOptions):
        """ Initialize network with composition approach """
        self.options: NetworkOptions = network_options

        # Sort nodes based on node-type
        self.options.nodes: list[NodeOptions] = sorted(
            self.options.nodes, key=lambda node: node["model"]
        )

        # Core network data and Cython implementation
        self.data: NetworkData = NetworkData.from_options(self.options)
        self.log: NetworkLog = NetworkLog.from_options(self.options)
        self.buffer_size: int = self.options.logs.buffer_size

        self.iteration = 0

        self._network_cy = NetworkCy(
            nnodes=len(self.options.nodes),
            nedges=len(self.options.edges),
            data=self.data,
            log=self.log
        )

        # Python-level collections
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

        # Setup the network
        self._setup_network()

        # Internal default solver
        self.solver: integrators.Integrator = None

        # Iteration
        if self.options.integration:
            self.timestep: float = self.options.integration.timestep
            self.iteration: int = 0
            self.n_iterations: int = self.options.integration.n_iterations
        else:
            raise ValueError("Integration options missing!")

    def step(self, time):
        """ Step the network using the internal integrator. """
        self.solver.step(
            self._network_cy,
            time,
            self.data.states.array
        )
        # Noise is updated via on_substep() inside the integrator,
        # using the correct sub-step size for adaptive integrators.

    def post_step(self, time, dt):
        """ Post-step housekeeping for external integrators.
        Call this after advancing states with an external integrator (e.g. scipy).
        Updates noise (Euler-Maruyama with timestep dt) and logs.

        Usage::

            scipy_solver.step()
            network.data.states.array[:] = scipy_solver.y
            network.post_step(time, dt)
        """
        self._network_cy.update_noise(time, dt)
        self.update_logs(time)

    def update_logs(self, time):
        """ Update logs for the current iteration """
        if self.options.logs.enable:
            buffer_iteration: int = (self.iteration % self.buffer_size)
            self._network_cy.update_logs(buffer_iteration, time)
        self.iteration += 1

    def run(self, n_iterations: Optional[int] = None):
        """ Run the network for n_iterations """
        if n_iterations is None:
            n_iterations = self.n_iterations

        for iteration in range(n_iterations):
            time = iteration * self.timestep
            self.step(time)
            self.update_logs(time)

    def _setup_network(self):
        """ Setup network nodes and edges """
        pylog.info(f"Number of nodes in network: {len(self.options.nodes)}")
        pylog.info(f"Number of edges in network: {len(self.options.edges)}")
        # Create Python nodes
        nstates = 0
        for index, node_options in enumerate(self.options.nodes):
            node_py = self._generate_node(node_options)
            node_py._node_cy.ninputs = len(
                self.data.connectivity.node_indices[
                    self.data.connectivity.index_offsets[index]:self.data.connectivity.index_offsets[index+1]
                ]
            ) if self.data.connectivity.index_offsets else 0
            nstates += node_py.nstates
            self.nodes.append(node_py)

        # Create Python edges
        for edge_options in self.options.edges:
            edge_py = self._generate_edge(edge_options)
            self.edges.append(edge_py)

        self._network_cy.nstates = nstates

        # Noise
        self.sde_noise = OrnsteinUhlenbeck(self.options)

        # Pass Python nodes/edges to Cython layer for C struct setup
        self._network_cy.setup_network(
            self.data, self.nodes, self.edges, self.sde_noise._ou_cy
        )

        # Initialize states
        self._initialize_states()

    def setup_integrator(self):
        """ Setup numerical integrators """
        self.solver = integrators.from_options(
            self.options.integration, self._network_cy.nstates
        )

    def _initialize_states(self):
        """ Initialize node states from options """
        for j, node_opts in enumerate(self.options.nodes):
            if node_opts.state:
                for state_index, index in enumerate(
                    range(self.data.states.indices[j], self.data.states.indices[j+1])
                ):
                    self.data.states.array[index] = node_opts.state.initial[state_index]

    @staticmethod
    def _generate_node(node_options: NodeOptions) -> Node:
        """ Generate a node from options """
        NodeClass = NodeFactory.create(node_options.model)
        return NodeClass.from_options(node_options)

    @staticmethod
    def _generate_edge(edge_options: EdgeOptions) -> Edge:
        """ Generate an edge from options """
        EdgeClass = EdgeFactory.create(edge_options.model)
        return EdgeClass.from_options(edge_options)

    def get_ode_func(self):
        """ Get ODE function for external integration """
        return self._network_cy.ode_func

    # Delegate properties to Cython implementation
    @property
    def nnodes(self) -> int:
        return self._network_cy.nnodes

    @property
    def nedges(self) -> int:
        return self._network_cy.nedges

    @property
    def nstates(self) -> int:
        return self._network_cy.nstates

    # Factory methods
    @classmethod
    def from_options(cls, options: NetworkOptions) -> Self:
        """ Initialize network from NetworkOptions """
        return cls(options)

    def to_options(self) -> NetworkOptions:
        """ Return NetworkOptions from network """
        return self.options
