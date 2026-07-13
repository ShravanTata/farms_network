""" Hopf Oscillator

[1]L. Righetti and A. J. Ijspeert, “Pattern generators with sensory
feedback for the control of quadruped locomotion,” in 2008 IEEE
International Conference on Robotics and Automation, May 2008,
pp. 819–824. doi: 10.1109/ROBOT.2008.4543306.
"""

from libc.stdio cimport printf


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    x = STATE_X
    y = STATE_Y



cdef void hopf_oscillator_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept:

    # States
    cdef double state_x = states[<int>STATE.x]
    cdef double state_y = states[<int>STATE.y]

    for j in range(inputs.ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]
        out.generic += (_weight*_input)


cdef void hopf_oscillator_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    # States
    cdef double state_x = states[<int>STATE.x]
    cdef double state_y = states[<int>STATE.y]

    cdef double input_val = input_vals.generic

    r_square = (state_x**2 + state_y**2)
    # xdot : x_dot
    derivatives[<int>STATE.x] = (
        params[<int>PARAM.alpha]*(params[<int>PARAM.mu] - r_square)*state_x - params[<int>PARAM.omega]*state_y
    )
    # ydot : y_dot
    derivatives[<int>STATE.y] = (
        params[<int>PARAM.beta]*(params[<int>PARAM.mu] - r_square)*state_y + params[<int>PARAM.omega]*state_x + (input_val)
    )


cdef double hopf_oscillator_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept:
    return states[<int>STATE.y]


cdef class HopfOscillatorNodeCy(NodeCy):
    """ Python interface to HopfOscillator Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.nstates = 2
        self._node.nparams = 4

        self._node.is_statefull = True
        self._node.input_tf = hopf_oscillator_input_tf
        self._node.ode = hopf_oscillator_ode
        self._node.output_tf = hopf_oscillator_output_tf

    def __init__(self):
        super().__init__()
