"""
Leaky Integrator Node Based on Danner et.al. with Na and K channels
"""

from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EXCITATORY, INHIBITORY, CHOLINERGIC


cdef enum:

    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_H = 1


cpdef enum PARAM:
    nparams = 19
    c_m = 0
    g_nap = 1
    e_na = 2
    v1_2_m = 3
    k_m = 4
    v1_2_h = 5
    k_h = 6
    v1_2_t = 7
    k_t = 8
    g_leak = 9
    e_leak = 10
    tau_0 = 11
    tau_max = 12
    v_max = 13
    v_thr = 14
    g_syn_e = 15
    g_syn_i = 16
    e_syn_e = 17
    e_syn_i = 18


cdef void li_nap_danner_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept


cdef void li_nap_danner_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double li_nap_danner_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class LINaPDannerNodeCy(NodeCy):
    """ Python interface to LI Danner NaP Node C-Structure """
