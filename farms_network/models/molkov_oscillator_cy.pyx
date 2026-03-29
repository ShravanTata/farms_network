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
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept:

    cdef molkov_oscillator_edge_params_t edge_params

    # States
    cdef double state_phase = states[<int>STATE.phase]

    # α from external input
    cdef double alpha = inputs.external_input

    cdef:
        double coupling = 0.0
        unsigned int j
        double _input, _weight, delta_phi
        double A, k
        unsigned int ninputs = inputs.ninputs

    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        edge_params = (<molkov_oscillator_edge_params_t*> edges[inputs.edge_indices[j]].params)[0]

        delta_phi = _input - state_phase

        # Compute speed-dependent coupling coefficients
        A = edge_params.a0 + edge_params.a1 * alpha
        k = (edge_params.k0 + edge_params.k1 * alpha) / edge_params.k_cpl

        coupling += k * (A * csin(delta_phi) - edge_params.b * csin(2.0 * delta_phi))

    out.phase_coupling = coupling


cdef void molkov_oscillator_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:

    # Parameters
    cdef molkov_oscillator_params_t* params = (<molkov_oscillator_params_t*> node[0].params)

    # dφ/dt = ω - coupling
    derivatives[<int>STATE.phase] = 2 * M_PI * params.intrinsic_frequency - input_vals.phase_coupling


cdef double molkov_oscillator_output_tf(
    double time,
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
        # parameters
        self.params = molkov_oscillator_params_t()
        self._node.params = <void*>&self.params
        if self._node.params is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, **kwargs):
        super().__init__()

        # Set node parameters
        self.params.intrinsic_frequency = kwargs.pop("intrinsic_frequency")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef molkov_oscillator_params_t params = (<molkov_oscillator_params_t*> self._node.params)[0]
        return params


cdef class MolkovOscillatorEdgeCy(EdgeCy):
    """ Python interface to Molkov Oscillator Edge C-Structure """

    def __cinit__(self, edge_type: str, **kwargs):
        # parameters
        self.params = molkov_oscillator_edge_params_t()
        self._edge.params = <void*>&self.params
        self._edge.nparams = 6
        if self._edge.params is NULL:
            raise MemoryError("Failed to allocate memory for edge parameters")

    def __init__(self, edge_type: str, **kwargs):
        super().__init__(edge_type)

        # Set edge parameters
        self.params.a0 = kwargs.pop("a0")
        self.params.a1 = kwargs.pop("a1")
        self.params.b = kwargs.pop("b")
        self.params.k0 = kwargs.pop("k0")
        self.params.k1 = kwargs.pop("k1")
        self.params.k_cpl = kwargs.pop("k_cpl")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef molkov_oscillator_edge_params_t params = (<molkov_oscillator_edge_params_t*> self._edge.params)[0]
        return params
