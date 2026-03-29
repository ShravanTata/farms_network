cimport numpy as cnp

from ..numeric.system_cy cimport ODESystemCy, SDESystemCy
from ..noise.ornstein_uhlenbeck_cy cimport OrnsteinUhlenbeckCy
from .data_cy cimport NetworkDataCy, NetworkLogCy
from .edge_cy cimport EdgeCy, edge_t
from .node_cy cimport NodeCy, node_t, node_inputs_t


cdef struct noise_t:
    # States
    int nstates
    double* states
    double* drift
    double* diffusion
    const unsigned int* indices
    # Outputs
    double* outputs


cdef struct network_t:
    # info
    unsigned int nnodes
    unsigned int nedges
    unsigned int nstates

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
    double* tmp_outputs

    double* external_inputs

    unsigned int* node_indices
    unsigned int* edge_indices
    double* weights
    unsigned int* index_offsets

    # Noise
    noise_t noise


cdef class NetworkCy(ODESystemCy):
    """ Python interface to Network ODE """

    cdef:
        network_t *_network
        public list nodes
        public list edges
        NetworkDataCy data
        NetworkLogCy log

        public unsigned int iteration
        const unsigned int n_iterations
        const unsigned int buffer_size
        const double timestep

        OrnsteinUhlenbeckCy sde_noise

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
    cdef void on_substep(self, double time, double h) noexcept
    cdef void c_update_noise(self, double time, double timestep) noexcept
    cdef void c_update_logs(self, unsigned int iteration, double time) noexcept
    # cpdef void update_iteration(self)


cdef class NetworkNoiseCy(SDESystemCy):
    """ Interface to stochastic noise in the network """

    cdef void evaluate_a(self, double time, double[:] states, double[:] drift) noexcept
    cdef void evaluate_b(self, double time, double[:] states, double[:] diffusion) noexcept
