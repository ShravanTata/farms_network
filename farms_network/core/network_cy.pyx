""" Network """

include "types.pxd"

import numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup, memcpy

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
    """ C implementation of the full network ODE evaluation.

    NOTE: Coupling uses explicit time-stepping — each node's input_tf reads
    outputs from the PREVIOUS evaluate() call. This introduces an O(dt)
    coupling lag, making the overall system first-order accurate for
    inter-node coupling regardless of integrator order. This is standard
    practice in computational neuroscience simulators (Brian2, NEST).
    For higher accuracy, reduce dt.
    """

    cdef node_t* __node
    cdef node_t** c_nodes = c_network.nodes
    cdef edge_t** c_edges = c_network.edges
    cdef unsigned int j
    cdef unsigned int nnodes = c_network.nnodes
    cdef processed_inputs_t processed_inputs
    cdef node_inputs_t node_inputs

    # It is important to use the states passed to the function and not from the data.states
    cdef double* states_ptr = &states[0]
    cdef double* derivatives_ptr = &derivatives[0]

    # Noise
    cdef double* noise = c_network.noise.outputs

    node_inputs.network_outputs = c_network.outputs

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
            <const node_t *> __node,
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
                <const node_t *> __node
            )

        # Compute output (for stateless nodes this is the actual output;
        # for stateful nodes this updates tmp_outputs for the post-evaluate swap)
        c_network.tmp_outputs[j] = __node.output_tf(
            time,
            <const double *> states_ptr + c_network.states_indices[j],
            processed_inputs,
            noise[j],
            <const node_t *> __node,
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


cdef inline void _init_data_pointers(network_t* net, NetworkDataCy data):
    """ Wire up network_t pointers from NetworkDataCy memoryviews. """

    # States
    if data.states.array.size > 0:
        net.states = &data.states.array[0]
        net.states_indices = &data.states.indices[0]
    else:
        net.states = NULL
        net.states_indices = NULL
    net.derivatives = NULL

    # Per-node arrays (required)
    net.external_inputs = &data.external_inputs.array[0]
    net.outputs = &data.outputs.array[0]
    net.tmp_outputs = &data.tmp_outputs.array[0]

    # Connectivity (optional — network may have 0 edges)
    net.node_indices = &data.connectivity.node_indices[0] if data.connectivity.node_indices.size > 0 else NULL
    net.edge_indices = &data.connectivity.edge_indices[0] if data.connectivity.edge_indices.size > 0 else NULL
    net.weights = &data.connectivity.weights[0] if data.connectivity.weights.size > 0 else NULL
    net.index_offsets = &data.connectivity.index_offsets[0] if data.connectivity.index_offsets.size > 0 else NULL

    # Noise (optional)
    net.noise.states = &data.noise.states[0] if data.noise.states.size > 0 else NULL
    net.noise.drift = &data.noise.drift[0] if data.noise.drift.size > 0 else NULL
    net.noise.diffusion = &data.noise.diffusion[0] if data.noise.diffusion.size > 0 else NULL
    net.noise.indices = &data.noise.indices[0] if data.noise.indices.size > 0 else NULL
    net.noise.outputs = &data.noise.outputs[0] if data.noise.outputs.size > 0 else NULL


cdef inline void _free_network(network_t* net) noexcept:
    """ Free all manually allocated memory on a network_t. """
    if net is NULL:
        return
    if net.nodes is not NULL:
        free(net.nodes)
    if net.edges is not NULL:
        free(net.edges)
    free(net)


cdef class NetworkCy(ODESystemCy):
    """ Network ODE """

    def __cinit__(self, nnodes: int, nedges: int, data: NetworkDataCy, log: NetworkLogCy):
        self._network = <network_t*>malloc(sizeof(network_t))
        if self._network is NULL:
            raise MemoryError("Failed to allocate memory for Network")

        self._network.nnodes = nnodes
        self._network.nedges = nedges
        self._network.nodes = <node_t**>malloc(nnodes * sizeof(node_t*))
        if self._network.nodes is NULL:
            raise MemoryError("Failed to allocate memory for Network nodes")
        self._network.edges = <edge_t**>malloc(nedges * sizeof(edge_t*)) if nedges > 0 else NULL

        self.data = <NetworkDataCy> data
        _init_data_pointers(self._network, self.data)

    def __init__(self, nnodes, nedges, data: NetworkDataCy, log: NetworkLogCy):
        """ Initialize """
        super().__init__()
        self.log = <NetworkLogCy>log
        self.iteration = 0

    def __dealloc__(self):
        _free_network(self._network)
        self._network = NULL

    def setup_network(
            self,
            data: NetworkData,
            nodes: List[NodeCy],
            edges: List[EdgeCy],
            sde_noise: SDESystemCy=None,
    ):
        """ Setup network """

        for index, node in enumerate(nodes):
            self._network.nodes[index] = <node_t*>((<NodeCy>node._node_cy)._node)

        for index, edge in enumerate(edges):
            self._network.edges[index] = <edge_t*>((<EdgeCy>edge._edge_cy)._edge)

        # Store the noise model
        self.sde_noise = sde_noise

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ Evaluate the ODE """
        ode(time, states, derivatives, self._network)
        # Copy new outputs from tmp buffer (memcpy avoids memoryview overhead)
        memcpy(
            self._network.outputs,
            self._network.tmp_outputs,
            self._network.nnodes * sizeof(double),
        )

    cdef void on_substep(self, double time, double h) noexcept:
        """ Called by integrators after each accepted sub-step.
        Advances the noise SDE by the sub-step size h. """
        self.c_update_noise(time, h)

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

    cdef void c_update_logs(self, unsigned int iteration, double time) noexcept:
        """ Copy current iteration data into logs using memcpy """
        cdef unsigned int nstates = self._network.nstates
        cdef unsigned int nnodes = self._network.nnodes

        self.log.times.array[iteration] = time

        if nstates > 0:
            memcpy(
                &self.log.states.array[iteration, 0],
                &self.data.states.array[0],
                nstates * sizeof(double),
            )
        memcpy(
            &self.log.external_inputs.array[iteration, 0],
            &self.data.external_inputs.array[0],
            nnodes * sizeof(double),
        )
        memcpy(
            &self.log.outputs.array[iteration, 0],
            &self.data.outputs.array[0],
            nnodes * sizeof(double),
        )

    def update_logs(self, iteration: int, time: float):
        """ Python wrapper for log update """
        self.c_update_logs(iteration, time)

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
