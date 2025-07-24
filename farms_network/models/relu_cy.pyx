""" Rectified Linear Unit """


from libc.stdio cimport printf
from libc.stdlib cimport free


cpdef enum STATE:

    #STATES
    nstates = NSTATES


cdef processed_inputs_t relu_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
) noexcept:
    cdef relu_params_t params = (<relu_params_t*> node[0].params)[0]
    cdef processed_inputs_t processed_inputs = {
        'generic': 0.0,
        'excitatory': 0.0,
        'inhibitory': 0.0,
        'cholinergic': 0.0,
        'phase_coupling': 0.0
    }

    cdef:
        double _sum = 0.0
        unsigned int j, ninputs
        double _input, _weight

    ninputs = inputs.ninputs

    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.source_indices[j]]
        _weight = inputs.weights[j]
        processed_inputs.generic += _weight*_input
    return processed_inputs


cdef void relu_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    raise NotImplementedError("ode must be implemented by node type")


cdef double relu_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    cdef relu_params_t params = (<relu_params_t*> node[0].params)[0]
    cdef double input_val = input_vals.generic
    cdef double res = max(0.0, params.gain*(params.sign*input_val + params.offset))
    return res


cdef class ReLUNodeCy(NodeCy):
    """ Python interface to ReLU Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.nstates = 0
        self._node.nparams = 3

        self._node.is_statefull = False
        self._node.input_tf = relu_input_tf
        self._node.output_tf = relu_output_tf
        # parameters
        self.params = relu_params_t()
        self._node.params = <void*>&self.params
        if self._node.params is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, **kwargs):
        super().__init__()

        # Set node parameters
        self.params.gain = kwargs.pop("gain")
        self.params.sign = kwargs.pop("sign")
        self.params.offset = kwargs.pop("offset")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def gain(self):
        """ Gain property """
        return (<relu_params_t*> self._node.params)[0].gain

    @gain.setter
    def gain(self, value):
        """ Set gain """
        (<relu_params_t*> self._node.params)[0].gain = value

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef relu_params_t params = (<relu_params_t*> self._node.params)[0]
        return params
