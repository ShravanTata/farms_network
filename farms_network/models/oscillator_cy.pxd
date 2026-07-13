""" Oscillator model """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EdgeCy


cdef enum:
    #STATES
    NSTATES = 3
    STATE_PHASE = 0
    STATE_AMPLITUDE= 1
    STATE_AMPLITUDE_0 = 2


cpdef enum PARAM:
    nparams = 9
    c_nu1 = 0         # frequency slope (Hz / a.u.)
    c_nu0 = 1         # frequency bias  (Hz)
    nu_sat = 2        # frequency outside drive range (Hz)
    c_R1 = 3          # amplitude slope (rad / a.u.)
    c_R0 = 4          # amplitude bias  (rad)
    R_sat = 5         # amplitude outside drive range (rad)
    d_low = 6         # lower bound of linear drive region (a.u.)
    d_high = 7        # upper bound of linear drive region (a.u.)
    amplitude_rate = 8


cpdef enum EDGE_PARAM:
    phase_difference = 0


cdef void oscillator_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept


cdef void oscillator_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double oscillator_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class OscillatorNodeCy(NodeCy):
    """ Python interface to Oscillator Node C-Structure """


cdef class OscillatorEdgeCy(EdgeCy):
    """ Python interface to Oscillator Edge C-Structure """
