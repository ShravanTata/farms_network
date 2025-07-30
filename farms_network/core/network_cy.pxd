cimport numpy as cnp

from ..numeric.integrators_cy cimport EulerMaruyamaSolver, RK4Solver
from ..numeric.system_cy cimport ODESystem, SDESystem
from .data_cy cimport NetworkDataCy, NodeDataCy
from .edge_cy cimport EdgeCy, edge_t
from .node_cy cimport NodeCy, node_t, node_inputs_t


cdef struct network_t:
    # info
    unsigned long int nnodes
    unsigned long int nedges
    unsigned long int nstates

    # nodes list
    node_t** nodes
    # edges list
    edge_t** edges

    # ODE
    double* states
    const unsigned int* states_indices

    double* derivatives
    const unsigned int* derivatives_indices

    double* outputs

    const double* external_inputs

    double* noise

    const unsigned int* node_indices
    const unsigned int* edge_indices
    const double* weights
    const unsigned int* index_offsets


cdef class NetworkCy(ODESystem):
    """ Python interface to Network ODE """

    cdef:
        network_t *_network
        public list nodes
        public list edges
        NetworkDataCy data

        unsigned int iteration
        const unsigned int n_iterations
        const unsigned int buffer_size
        const double timestep

    cpdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
    cpdef void step(self)
