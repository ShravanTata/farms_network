""" Molkov oscillator model (Molkov et al. 2015)

Phase oscillator with speed-dependent coupling for interlimb coordination.
ODE: dφ/dt = ω - Σ_j k(α) * [A(α)*sin(Δφ) - B*sin(2Δφ)]
where A(α) = a0 + a1*α, k(α) = (k0 + k1*α) / k_cpl
"""


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EdgeCy


cdef enum:
    #STATES
    NSTATES = 1
    STATE_PHASE = 0


cpdef enum PARAM:
    nparams = 1
    intrinsic_frequency = 0


cpdef enum EDGE_PARAM:
    a0 = 0
    a1 = 1
    b = 2
    k0 = 3
    k1 = 4
    k_cpl = 5


cdef void molkov_oscillator_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept


cdef void molkov_oscillator_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double molkov_oscillator_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class MolkovOscillatorNodeCy(NodeCy):
    """ Python interface to Molkov Oscillator Node C-Structure """


cdef class MolkovOscillatorEdgeCy(EdgeCy):
    """ Python interface to Molkov Oscillator Edge C-Structure """
