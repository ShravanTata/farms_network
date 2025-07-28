""" Network """

include "types.pxd"

import numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from ..models.factory import NodeFactory
from ..noise.ornstein_uhlenbeck import OrnsteinUhlenbeck
from .data import NetworkData, NetworkStates
from .node_cy cimport processed_inputs_t

from .data_cy cimport (NetworkConnectivityCy, NetworkDataCy, NetworkNoiseCy,
                       NetworkStatesCy)

from typing import List

from .options import (EdgeOptions, IntegrationOptions, NetworkOptions,
                      NodeOptions)


cdef inline void ode(
    double time,
    double[:] states_arr,
    network_t* c_network,
    double[:] node_outputs_tmp,
) noexcept:
    """ C Implementation to compute full network state """
    cdef unsigned int j, nnodes

    cdef node_t __node
    cdef node_t** c_nodes = c_network.nodes
    cdef edge_t** c_edges = c_network.edges
    nnodes = c_network.nnodes

    cdef double* node_outputs_tmp_ptr = &node_outputs_tmp[0]

    cdef node_inputs_t node_inputs
    node_inputs.network_outputs = c_network.outputs
    # It is important to use the states passed to the function and not from the data.states
    c_network.states = &states_arr[0]

    cdef processed_inputs_t processed_inputs

    for j in range(nnodes):
        total_input_val = 0.0
        __node = c_nodes[j][0]
        # Prepare node context
        node_inputs.source_indices = c_network.input_neurons + c_network.input_neurons_indices[j]
        node_inputs.weights = c_network.weights + c_network.input_neurons_indices[j]
        node_inputs.external_input = c_network.external_inputs[j]

        node_inputs.ninputs = __node.ninputs
        node_inputs.node_index = j
        if __node.is_statefull:
            # Compute the inputs from all nodes
            processed_inputs = __node.input_tf(
                time,
                c_network.states + c_network.states_indices[j],
                node_inputs,
                c_nodes[j],
                c_edges + c_network.input_neurons_indices[j],
            )
            # Compute the ode
            __node.ode(
                time,
                c_network.states + c_network.states_indices[j],
                c_network.derivatives + c_network.derivatives_indices[j],
                processed_inputs,
                0.0,
                c_nodes[j]
            )
            # Compute all the node outputs based on the current state
            node_outputs_tmp_ptr[j] = __node.output_tf(
                time,
                c_network.states + c_network.states_indices[j],
                processed_inputs,
                0.0,
                c_nodes[j],
            )
        else:
            processed_inputs = __node.input_tf(
                time,
                NULL,
                node_inputs,
                c_nodes[j],
                c_edges + c_network.input_neurons_indices[j],
            )
            # Compute all the node outputs based on the current state
            node_outputs_tmp_ptr[j] = __node.output_tf(
                time,
                NULL,
                processed_inputs,
                0.0,
                c_nodes[j],
            )


# cdef inline void logger(
#     int iteration,
#     NetworkDataCy data,
#     network_t* c_network
# ) noexcept:
#     cdef unsigned int nnodes = c_network.nnodes
#     cdef unsigned int j
#     cdef double* states_ptr = &data.states.array[0]
#     cdef unsigned int[:] state_indices = data.states.indices
#     cdef double[:] outputs = data.outputs.array
#     cdef double* outputs_ptr = &data.outputs.array[0]
#     cdef double[:] external_inputs = data.external_inputs.array
#     cdef NodeDataCy node_data
#     cdef double[:] node_states
#     cdef int state_idx, start_idx, end_idx, state_iteration
#     cdef NodeDataCy[:] nodes_data = data.nodes
#     for j in range(nnodes):
#         # Log states
#         start_idx = state_indices[j]
#         end_idx = state_indices[j+1]
#         state_iteration = 0
#         node_states = nodes_data[j].states.array[iteration]
#         for state_idx in range(start_idx, end_idx):
#             node_states[state_iteration] = states_ptr[state_idx]
#             state_iteration += 1
#         nodes_data[j].output.array[iteration] = outputs_ptr[j]
#         nodes_data[j].external_input.array[iteration] = external_inputs[j]


# cdef inline void _noise_states_to_output(
#     double[:] states,
#     unsigned int[:] indices,
#     double[:] outputs,
# ) noexcept:
#     """ Copy noise states data to noise outputs """
#     cdef int n_indices = indices.shape[0]
#     cdef int index
#     for index in range(n_indices):
#         outputs[indices[index]] = states[index]


cdef class NetworkCy(ODESystem):
    """ Python interface to Network ODE """

    def __cinit__(self, nnodes: int, nedges: int, data: NetworkDataCy):
        # Memory allocation only
        self._network = <network_t*>malloc(sizeof(network_t))
        if self._network is NULL:
            raise MemoryError("Failed to allocate memory for Network")

        self._network.nnodes = nnodes
        self._network.nedges = nedges
        self.__tmp_node_outputs = np.zeros((nnodes,))

        # Allocate C arrays
        self._network.nodes = <node_t**>malloc(self.nnodes * sizeof(node_t*))
        if self._network.nodes is NULL:
            raise MemoryError("Failed to allocate memory for Network nodes")
        self._network.edges = <edge_t**>malloc(self.nedges * sizeof(edge_t*))
        if self._network.edges is NULL:
            raise MemoryError("Failed to allocate memory for Network edges")

        # Initialize network context
        self.data = <NetworkDataCy> data
        if self.data.states.array.size > 0:
            self._network.states = &self.data.states.array[0][0]
        else:
            self._network.states = NULL  # No stateful

        if self.data.states.indices.size > 0:
            self._network.states_indices = &self.data.states.indices[0]
        else:
            self._network.states_indices = NULL

        if self.data.derivatives.array.size > 0:
            self._network.derivatives = &self.data.derivatives.array[0][0]
        else:
            self._network.derivatives = NULL

        if self.data.derivatives.indices.size > 0:
            self._network.derivatives_indices = &self.data.derivatives.indices[0]
        else:
            self._network.derivatives_indices = NULL

        if self.data.external_inputs.array.size > 0:
            self._network.external_inputs = &self.data.external_inputs.array[0][0]
        else:
            self._network.external_inputs = NULL

        if self.data.outputs.array.size > 0:
            self._network.outputs = &self.data.outputs.array[0][0]
        else:
            self._network.outputs = NULL

        if self.data.noise.outputs.size > 0:
            self._network.noise = &self.data.noise.outputs[0]
        else:
            self._network.noise = NULL

        if self.data.connectivity.sources.size > 0:
            self._network.input_neurons = &self.data.connectivity.sources[0]
        else:
            self._network.input_neurons = NULL

        if self.data.connectivity.weights.size > 0:
            self._network.weights = &self.data.connectivity.weights[0]
        else:
            self._network.weights = NULL

        if self.data.connectivity.indices.size > 0:
            self._network.input_neurons_indices = &self.data.connectivity.indices[0]
            print("indices size", self.data.connectivity.indices.size)
        else:
            self._network.input_neurons_indices = NULL

        # cdef double* node_outputs_tmp_ptr = &node_outputs_tmp[0]

    def __init__(self, nnodes, nedges, data: NetworkDataCy):
        """ Initialize """
        super().__init__()
        self.iteration = 0


    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        if self._network.nodes is not NULL:
            free(self._network.nodes)
        if self._network.edges is not NULL:
            free(self._network.edges)
        if self._network is not NULL:
            free(self._network)

    def setup_network(self, options: NetworkOptions, data: NetworkData, nodes: List[NodeCy], edges: List[EdgeCy]):
        """ Setup network """

        for index, node in enumerate(nodes):
            self._network.nodes[index] = <node_t*>((<NodeCy>node._node_cy)._node)

        for index, edge in enumerate(edges):
            self._network.edges[index] = <edge_t*>((<EdgeCy>edge._edge_cy)._edge)

    @property
    def nnodes(self):
        """ Number of nodes in the network """
        return self._network.nnodes

    @property
    def nedges(self):
        """ Number of edges in the network """
        return self._network.nedges

    @property
    def nstates(self):
        """ Number of states in the network """
        return self._network.nstates

    @nstates.setter
    def nstates(self, value: int):
        """ Number of network states """
        self._network.nstates = value

    cpdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ Evaluate the ODE """
        # Update noise model
        cdef NetworkDataCy data = <NetworkDataCy> self.data

        ode(time, states, self._network, self.__tmp_node_outputs)
        data.states.array[self.iteration, :] = states
        data.outputs.array[self.iteration, :] = self.__tmp_node_outputs
        derivatives[:] = data.derivatives.array[self.iteration, :]

    cpdef void step(self):
        """ Step the network state """
        cdef NetworkDataCy data = self.data
        cdef SDESystem sde_system = self.sde_system
        cdef EulerMaruyamaSolver sde_integrator = self.sde_integrator

        # sde_integrator.step(
        #     sde_system,
        #     (self.iteration%self.buffer_size)*self.timestep,
        #     self.data.noise.states
        # )
        # _noise_states_to_output(
        #     self.data.noise.states,
        #     self.data.noise.indices,
        #     self.data.noise.outputs
        # )
        # self.ode_integrator.step(
        #     self,
        #     (self.iteration%self.buffer_size)*self.timestep,
        #     self.data.states.array
        # )
        # Logging
        # TODO: Use network options to check global logging flag
        # logger((self.iteration%self.buffer_size), self.data, self._network)
        self.iteration += 1

    def setup_integrator(self, network_options: NetworkOptions):
        """ Setup integrator for neural network """
        # Setup ODE numerical integrator
        integration_options = network_options.integration
        timestep = integration_options.timestep
        self.ode_integrator = RK4Solver(self._network.nstates, timestep)
        # Setup SDE numerical integrator for noise models if any
        noise_options = []
        for node in network_options.nodes:
            if node.noise is not None:
                if node.noise.is_stochastic:
                    noise_options.append(node.noise)

        self.sde_system = OrnsteinUhlenbeck(noise_options)
        self.sde_integrator = EulerMaruyamaSolver(len(noise_options), timestep)
