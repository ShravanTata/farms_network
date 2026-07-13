""" Molkov oscillator model (Molkov et al. 2015)

Phase oscillator with speed-dependent coupling for interlimb coordination.
"""


from libc.math cimport M_PI
from libc.math cimport sin as csin


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    phase = STATE_PHASE



cdef void molkov_oscillator_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept:

    # States
    cdef double state_phase = states[<int>STATE.phase]

    # α from external input
    cdef double alpha = inputs.external_input

    cdef:
        double coupling = 0.0
        unsigned int j
        double _input, delta_phi
        double A, k
        const double* ep
        unsigned int ninputs = inputs.ninputs

    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        ep = inputs.edge_params + inputs.edge_params_indices[inputs.edge_indices[j]]

        delta_phi = _input - state_phase

        # Compute speed-dependent coupling coefficients
        A = ep[<int>EDGE_PARAM.a0] + ep[<int>EDGE_PARAM.a1] * alpha
        k = (ep[<int>EDGE_PARAM.k0] + ep[<int>EDGE_PARAM.k1] * alpha) / ep[<int>EDGE_PARAM.k_cpl]

        coupling += k * (A * csin(delta_phi) - ep[<int>EDGE_PARAM.b] * csin(2.0 * delta_phi))

    out.phase_coupling = coupling


cdef void molkov_oscillator_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:

    # dφ/dt = ω - coupling
    derivatives[<int>STATE.phase] = 2 * M_PI * params[<int>PARAM.intrinsic_frequency] - input_vals.phase_coupling


cdef double molkov_oscillator_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept:
    return states[<int>STATE.phase]


cdef class MolkovOscillatorNodeCy(NodeCy):
    """ Python interface to Molkov Oscillator Node C-Structure """

    def __cinit__(self):
        self._node.nstates = 1
        self._node.nparams = 1

        self._node.is_statefull = True
        self._node.input_tf = molkov_oscillator_input_tf
        self._node.ode = molkov_oscillator_ode
        self._node.output_tf = molkov_oscillator_output_tf

    def __init__(self):
        super().__init__()


cdef class MolkovOscillatorEdgeCy(EdgeCy):
    """ Python interface to Molkov Oscillator Edge C-Structure """

    def __cinit__(self, edge_type: str):
        pass

    def __init__(self, edge_type: str):
        super().__init__(edge_type)
