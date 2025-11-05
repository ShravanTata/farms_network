""" Network """

include "types.pxd"

import numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from ..models.factory import NodeFactory

from libc.math cimport sqrt as csqrt

from ..noise.ornstein_uhlenbeck_cy cimport OrnsteinUhlenbeckCy

from .data import NetworkData, NetworkStates

from .data_cy cimport (NetworkConnectivityCy, NetworkDataCy, NetworkNoiseCy,
                       NetworkStatesCy)
from .node_cy cimport processed_inputs_t

from typing import List

from .options import NetworkOptions


cdef inline void ode(
    double time,
    double[:] states,
    double[:] derivatives,
    network_t* c_network,
) noexcept:
    """ C Implementation to compute full network state """

    cdef node_t* __node
    cdef node_t** c_nodes = c_network.nodes
    cdef edge_t** c_edges = c_network.edges
    cdef unsigned int nnodes = c_network.nnodes
    cdef unsigned int j
    cdef processed_inputs_t processed_inputs = {
        'generic': 0.0,
        'excitatory': 0.0,
        'inhibitory': 0.0,
        'cholinergic': 0.0,
        'phase_coupling': 0.0
    }
    cdef node_inputs_t node_inputs

    node_inputs.network_outputs = c_network.outputs
    # It is important to use the states passed to the function and not from the data.states
    cdef double* states_ptr = &states[0]
    cdef double* derivatives_ptr = &derivatives[0]

    # Noise
    cdef double* noise = c_network.noise.outputs

    for j in range(nnodes):
        __node = c_nodes[j]

        # Prepare node context
        node_inputs.node_indices = c_network.node_indices + c_network.index_offsets[j]
        node_inputs.edge_indices = c_network.edge_indices + c_network.index_offsets[j]
        node_inputs.weights = c_network.weights + c_network.index_offsets[j]
        node_inputs.external_input = c_network.external_inputs[j]

        node_inputs.ninputs = __node.ninputs
        node_inputs.node_index = j

        # Compute the inputs from all nodes
        processed_inputs.generic = 0.0
        processed_inputs.excitatory = 0.0
        processed_inputs.inhibitory = 0.0
        processed_inputs.cholinergic = 0.0
        processed_inputs.phase_coupling = 0.0

        __node.input_tf(
            time,
            states_ptr + c_network.states_indices[j],
            <const node_inputs_t> node_inputs,
            <const node_t *> c_nodes[j],
            <const edge_t **> c_edges,
            &processed_inputs
        )

        if __node.is_statefull:
            # Compute the ode
            __node.ode(
                time,
                <const double *> states_ptr + c_network.states_indices[j],
                derivatives_ptr + c_network.states_indices[j],
                processed_inputs,
                noise[j],
                <const node_t *> c_nodes[j]
            )
        # Check for writing to proper outputs array
        c_network.tmp_outputs[j] = __node.output_tf(
            time,
            <const double *> states_ptr + c_network.states_indices[j],
            processed_inputs,
            noise[j],
            <const node_t *> c_nodes[j],
        )


cdef inline void _noise_states_to_output(
    double[:] states,
    unsigned int[:] indices,
    double[:] outputs,
) noexcept:
    """ Copy noise states data to noise outputs """
    cdef int n_indices = indices.shape[0]
    cdef int index
    for index in range(n_indices):
        outputs[indices[index]] = states[index]


cdef class NetworkCy(ODESystem):
    """ Python interface to Network ODE """

    def __cinit__(self, nnodes: int, nedges: int, data: NetworkDataCy, log: NetworkLogCy):
        # Memory allocation only
        self._network = <network_t*>malloc(sizeof(network_t))
        if self._network is NULL:
            raise MemoryError("Failed to allocate memory for Network")

        self._network.nnodes = nnodes
        self._network.nedges = nedges

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
            self._network.states = &self.data.states.array[0]
        else:
            self._network.states = NULL  # No stateful

        if self.data.states.indices.size > 0:
            self._network.states_indices = &self.data.states.indices[0]
        else:
            assert self._network.states == NULL
            self._network.states_indices = NULL

        # if self.data.derivatives.array.size > 0:
        #     self._network.derivatives = &self.data.derivatives.array[0]
        # else:
        #     assert self._network.states == NULL
        self._network.derivatives = NULL

        # if self.data.derivatives.indices.size > 0:
        #     self._network.derivatives_indices = &self.data.derivatives.indices[0]
        # else:
        #     assert self._network.derivatives == NULL
        self._network.derivatives_indices = NULL

        if self.data.external_inputs.array.size > 0:
            self._network.external_inputs = &self.data.external_inputs.array[0]
        else:
            raise ValueError("External inputs array cannot be of size 0")

        if self.data.outputs.array.size > 0:
            self._network.outputs = &self.data.outputs.array[0]
        else:
            raise ValueError("Outputs array cannot be of size 0")

        if self.data.tmp_outputs.array.size > 0:
            self._network.tmp_outputs = &self.data.tmp_outputs.array[0]
        else:
            raise ValueError("Temp Outputs array cannot be of size 0")

        if self.data.connectivity.node_indices.size > 0:
            self._network.node_indices = &self.data.connectivity.node_indices[0]
        else:
            raise ValueError("Connectivity array cannot be of size 0")

        if self.data.connectivity.edge_indices.size > 0:
            self._network.edge_indices = &self.data.connectivity.edge_indices[0]
        else:
            raise ValueError("Connectivity array cannot be of size 0")

        if self.data.connectivity.weights.size > 0:
            self._network.weights = &self.data.connectivity.weights[0]
        else:
            raise ValueError("Connectivity array cannot be of size 0")

        if self.data.connectivity.index_offsets.size > 0:
            self._network.index_offsets = &self.data.connectivity.index_offsets[0]
        else:
            raise ValueError("Connectivity array cannot be of size 0")

        # Noise
        if self.data.noise.states.size > 0:
            self._network.noise.states = &self.data.noise.states[0]
        else:
            self._network.noise.states = NULL

        if self.data.noise.drift.size > 0:
            self._network.noise.drift = &self.data.noise.drift[0]
        else:
            self._network.noise.drift = NULL

        if self.data.noise.diffusion.size > 0:
            self._network.noise.diffusion = &self.data.noise.diffusion[0]
        else:
            self._network.noise.diffusion = NULL

        if self.data.noise.indices.size > 0:
            self._network.noise.indices = &self.data.noise.indices[0]
        else:
            self._network.noise.indices = NULL

        if self.data.noise.outputs.size > 0:
            self._network.noise.outputs = &self.data.noise.outputs[0]
        else:
            self._network.noise.outputs = NULL


    def __init__(self, nnodes, nedges, data: NetworkDataCy, log: NetworkLogCy):
        """ Initialize """
        super().__init__()
        self.log = <NetworkLogCy>log
        self.iteration = 0

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        if self._network.nodes is not NULL:
            free(self._network.nodes)
            self._network.nodes = NULL
        if self._network.edges is not NULL:
            free(self._network.edges)
            self._network.edges = NULL
        if self._network is not NULL:
            free(self._network)
            self._network = NULL

    def setup_network(
            self,
            data: NetworkData,
            nodes: List[NodeCy],
            edges: List[EdgeCy],
            sde_noise: SDESystem=None,
    ):
        """ Setup network """

        for index, node in enumerate(nodes):
            self._network.nodes[index] = <node_t*>((<NodeCy>node._node_cy)._node)

        for index, edge in enumerate(edges):
            self._network.edges[index] = <edge_t*>((<EdgeCy>edge._edge_cy)._edge)

        self.sde_noise = sde_noise

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ Evaluate the ODE """
        # Update network ODE
        ode(time, states, derivatives, self._network)
        # Swap the temporary outputs
        self.data.outputs.array[:] = self.data.tmp_outputs.array[:]

    cdef void c_update_noise(self, double time, double timestep) noexcept:
        """ Update """
        if self.sde_noise is not None:
            self.sde_noise.evaluate_a(time, self.data.noise.states, self.data.noise.drift)
            self.sde_noise.evaluate_b(time, self.data.noise.states, self.data.noise.diffusion)
            for j in range(self.sde_noise.n_dim):
                self.data.noise.states[j] += (
                    self.data.noise.drift[j]*timestep + csqrt(timestep)*self.data.noise.diffusion[j]
                )
                self.data.noise.outputs[self.data.noise.indices[j]] = self.data.noise.states[j]

    def update_noise(self, double time, double timestep):
        self.c_update_noise(time, timestep)

    def ode_func(self, double time, double[:] states):
        """ Evaluate the ODE """
        self.evaluate(time, states, self.data.derivatives.array)
        return self.data.derivatives.array

    def update_logs(self, time: float):
        """ Updated logs to copy current iteration data into logs """
        self.iteration += 1
        self.log.times.array[self.iteration] = time
        self.log.states.array[self.iteration, :] = self.data.states.array[:]
        self.log.external_inputs.array[self.iteration, :] = self.data.external_inputs.array[:]
        self.log.outputs.array[self.iteration, :] = self.data.outputs.array[:]

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

    @property
    def noise_nstates(self):
        """ Number of noise states in the network """
        return self._network.noise.nstates

    @noise_nstates.setter
    def noise_nstates(self, value: int):
        """ Number of network noise states """
        self._network.noise.nstates = value
