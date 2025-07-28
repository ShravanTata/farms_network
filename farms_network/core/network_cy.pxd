cimport numpy as cnp

from ..numeric.integrators_cy cimport EulerMaruyamaSolver, RK4Solver
from ..numeric.system cimport ODESystem, SDESystem
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
    unsigned int* states_indices

    double* derivatives
    unsigned int* derivatives_indices

    double* outputs

    double* external_inputs

    double* noise

    unsigned int* input_neurons
    double* weights
    unsigned int* input_neurons_indices


cdef class NetworkCy(ODESystem):
    """ Python interface to Network ODE """

    cdef:
        network_t *_network
        public list nodes
        public list edges
        public NetworkDataCy data
        double[:] __tmp_node_outputs

        unsigned int iteration
        unsigned int n_iterations
        unsigned int buffer_size
        double timestep

        public RK4Solver ode_integrator
        public EulerMaruyamaSolver sde_integrator

        SDESystem sde_system

        list nodes_output_data

    cpdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
    cpdef void step(self)
