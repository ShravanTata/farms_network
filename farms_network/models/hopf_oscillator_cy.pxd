""" Hopf-Oscillator model """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t


cdef enum:
    #STATES
    NSTATES = 2
    STATE_X = 0
    STATE_Y= 1


cpdef enum PARAM:
    nparams = 4
    mu = 0
    omega = 1
    alpha = 2
    beta = 3


cdef void hopf_oscillator_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept


cdef void hopf_oscillator_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double hopf_oscillator_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class HopfOscillatorNodeCy(NodeCy):
    """ Python interface to HopfOscillator Node C-Structure """
