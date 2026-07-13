from libc.math cimport cosh as ccosh
from libc.math cimport exp as cexp
from libc.math cimport fabs as cfabs
from libc.stdio cimport printf
from libc.string cimport strdup
import numpy as np

from farms_network.models import Models


cpdef enum STATE:
    #STATES
    nstates = NSTATES
    v = STATE_V
    h = STATE_H


cdef void li_nap_danner_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept:

    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_h = states[<int>STATE.h]

    # Neuron inputs
    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight
        edge_t* _edge

    cdef unsigned int ninputs = inputs.ninputs
    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]
        _edge = edges[inputs.edge_indices[j]]
        if _edge.type == EXCITATORY:
            # Excitatory Synapse
            out.excitatory += params[<int>PARAM.g_syn_e]*cfabs(_weight)*_input*(state_v - params[<int>PARAM.e_syn_e])
        elif _edge.type == INHIBITORY:
            # Inhibitory Synapse
            out.inhibitory += params[<int>PARAM.g_syn_i]*cfabs(_weight)*_input*(state_v - params[<int>PARAM.e_syn_i])


cdef void li_nap_danner_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_h = states[<int>STATE.h]

    # tau_h(V)
    cdef double tau_h = params[<int>PARAM.tau_0] + (params[<int>PARAM.tau_max] - params[<int>PARAM.tau_0]) / \
        ccosh((state_v - params[<int>PARAM.v1_2_t]) / params[<int>PARAM.k_t])

    # h_inf(V)
    cdef double h_inf = 1./(1.0 + cexp((state_v - params[<int>PARAM.v1_2_h]) / params[<int>PARAM.k_h]))

    # m(V)
    cdef double m = 1./(1.0 + cexp((state_v - params[<int>PARAM.v1_2_m]) / params[<int>PARAM.k_m]))

    # Inap
    cdef double i_nap = params[<int>PARAM.g_nap] * m * state_h * (state_v - params[<int>PARAM.e_na])

    # Ileak
    cdef double i_leak = params[<int>PARAM.g_leak] * (state_v - params[<int>PARAM.e_leak])

    # noise current
    cdef double i_noise = noise

    # Slow inactivation
    derivatives[<int>STATE.h] = (h_inf - state_h) / tau_h

    # dV
    derivatives[<int>STATE.v] = -(
        i_nap + i_leak + i_noise + input_vals.excitatory + input_vals.inhibitory
    )/params[<int>PARAM.c_m]


cdef double li_nap_danner_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    cdef double _n_out = 0.0
    cdef double state_v = states[<int>STATE.v]
    if state_v >= params[<int>PARAM.v_max]:
        _n_out = 1.0
    elif (params[<int>PARAM.v_thr] <= state_v) and (state_v < params[<int>PARAM.v_max]):
        _n_out = (state_v - params[<int>PARAM.v_thr]) / (params[<int>PARAM.v_max] - params[<int>PARAM.v_thr])
    elif state_v < params[<int>PARAM.v_thr]:
        _n_out = 0.0
    return _n_out


cdef class LINaPDannerNodeCy(NodeCy):
    """ Python interface to LI Danner NaP Node C-Structure """

    def __cinit__(self):
        self._node.nstates = 2
        self._node.nparams = 19

        self._node.is_statefull = True

        self._node.input_tf = li_nap_danner_input_tf
        self._node.ode = li_nap_danner_ode
        self._node.output_tf = li_nap_danner_output_tf

    def __init__(self):
        super().__init__()
