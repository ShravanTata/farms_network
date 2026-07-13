""" Linear model """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t


cdef enum:
    #STATES
    NSTATES = 0


cpdef enum PARAM:
    nparams = 2
    slope = 0
    bias = 1


cdef void linear_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept


cdef void linear_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept


cdef double linear_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept


cdef class LinearNodeCy(NodeCy):
    """ Python interface to Linear Node C-Structure """
