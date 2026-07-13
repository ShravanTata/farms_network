""" Linear model """

from libc.stdio cimport printf
from libc.stdlib cimport free


cpdef enum STATE:
    #STATES
    nstates = NSTATES


cdef void linear_input_tf(
    double time,
    const double* params,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out,
) noexcept:
    cdef:
        double _sum = 0.0
        unsigned int j, ninputs
        double _input, _weight

    ninputs = inputs.ninputs

    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]
        out.generic += _weight*_input


cdef void linear_ode(
    double time,
    const double* params,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    raise NotImplementedError("ode must be implemented by node type")


cdef double linear_output_tf(
    double time,
    const double* params,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    cdef double input_val = input_vals.generic
    cdef double res = params[<int>PARAM.slope]*input_val + params[<int>PARAM.bias]
    return res


cdef class LinearNodeCy(NodeCy):
    """ Python interface to Linear Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.nstates = 0
        self._node.nparams = 2

        self._node.is_statefull = False
        self._node.input_tf = linear_input_tf
        self._node.output_tf = linear_output_tf

    def __init__(self):
        super().__init__()
