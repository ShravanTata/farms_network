""" Leaky Integrator Node Based on Danner et.al. 2016 """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EXCITATORY, INHIBITORY, CHOLINERGIC


cdef enum:
    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_A = 1


cpdef enum PARAM:
    nparams = 10
    c_m = 0
    g_leak = 1
    e_leak = 2
    v_max = 3
    v_thr = 4
    g_syn_e = 5
    g_syn_i = 6
    e_syn_e = 7
    e_syn_i = 8
    tau_ch = 9


cdef void li_danner_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept


cdef void li_danner_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double li_danner_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class LIDannerNodeCy(NodeCy):
    """ Python interface to LI Danner Node C-Structure """
