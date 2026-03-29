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


cdef packed struct molkov_oscillator_params_t:

    double intrinsic_frequency     # Hz (ω)


cdef packed struct molkov_oscillator_edge_params_t:

    double a0          # A(α) = a0 + a1*α
    double a1
    double b           # B (constant)
    double k0          # k(α) = (k0 + k1*α) / k_cpl
    double k1
    double k_cpl       # coupling normalization


cdef void molkov_oscillator_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept


cdef void molkov_oscillator_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double molkov_oscillator_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class MolkovOscillatorNodeCy(NodeCy):
    """ Python interface to Molkov Oscillator Node C-Structure """

    cdef:
        molkov_oscillator_params_t params


cdef class MolkovOscillatorEdgeCy(EdgeCy):
    """ Python interface to Molkov Oscillator Edge C-Structure """

    cdef:
        molkov_oscillator_edge_params_t params
