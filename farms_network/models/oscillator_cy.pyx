""" Oscillator model """


from libc.math cimport M_PI
from libc.math cimport sin as csin
from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from ..core.edge_cy cimport GENERIC


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    phase = STATE_PHASE
    amplitude = STATE_AMPLITUDE
    amplitude_0 = STATE_AMPLITUDE_0


cdef void oscillator_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept:

    # States of the current (target) node
    cdef double state_phase = states[<int>STATE.phase]

    cdef:
        unsigned int j
        double _input, _weight, _r_j
        const double* ep
        unsigned int edge_type
        unsigned int ninputs = inputs.ninputs

    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]
        edge_type = edges[inputs.edge_indices[j]][0].type

        if edge_type == GENERIC:
            # Drive signal d accumulates in out.generic
            out.generic += _weight * _input
        else:
            # Phase coupling: Eq 1 of Ijspeert 2007
            # Uses r_j (source amplitude), not r_i (target)
            ep = inputs.edge_params + inputs.edge_params_indices[inputs.edge_indices[j]]
            _r_j = inputs.network_states[
                inputs.states_indices[inputs.node_indices[j]] + <int>STATE.amplitude
            ]
            out.phase_coupling += _weight * _r_j * csin(
                _input - state_phase - ep[<int>EDGE_PARAM.phase_difference]
            )


cdef void oscillator_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:

    # States
    cdef double state_amplitude = states[<int>STATE.amplitude]
    cdef double state_amplitude_0 = states[<int>STATE.amplitude_0]

    # Drive signal d → frequency ν(d) and nominal amplitude R(d)
    cdef double d = input_vals.generic
    cdef double nu, R

    if params[<int>PARAM.d_low] <= d <= params[<int>PARAM.d_high]:
        nu = params[<int>PARAM.c_nu1] * d + params[<int>PARAM.c_nu0]
        R  = params[<int>PARAM.c_R1]  * d + params[<int>PARAM.c_R0]
    else:
        nu = params[<int>PARAM.nu_sat]
        R  = params[<int>PARAM.R_sat]

    derivatives[<int>STATE.phase] = 2*M_PI*nu + input_vals.phase_coupling
    derivatives[<int>STATE.amplitude] = state_amplitude_0
    derivatives[<int>STATE.amplitude_0] = params[<int>PARAM.amplitude_rate]*(
        (params[<int>PARAM.amplitude_rate]/4.0)*(R - state_amplitude) - state_amplitude_0
    )


cdef double oscillator_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept:
    return states[<int>STATE.phase]


cdef class OscillatorNodeCy(NodeCy):
    """ Python interface to Oscillator Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.nstates = 3
        self._node.nparams = 9

        self._node.is_statefull = True
        self._node.input_tf = oscillator_input_tf
        self._node.ode = oscillator_ode
        self._node.output_tf = oscillator_output_tf

    def __init__(self):
        super().__init__()


cdef class OscillatorEdgeCy(EdgeCy):
    """ Python interface to Oscillator Edge C-Structure """

    def __cinit__(self, edge_type: str):
        pass

    def __init__(self, edge_type: str):
        super().__init__(edge_type)
