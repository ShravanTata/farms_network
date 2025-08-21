""" Network Data """


from farms_core.array.array_cy cimport (DoubleArray1D, DoubleArray2D, IntegerArray1D)

include 'types.pxd'


# cdef class NetworkComputeData:

#     cdef:
#         # States
#         public DoubleArray1D curr_states
#         public DoubleArray1D tmp_states
#         public UITYPEv1 state_indices
#         # Derivatives
#         public DoubleArray1D curr_derivatives
#         public DoubleArray1D tmp_derivatives
#         # Outputs
#         public DoubleArray1D curr_outputs
#         public DoubleArray1D tmp_outputs
#         # External inputs
#         public DoubleArray2D external_inputs
#         # Network connectivity
#         public NetworkConnectivityCy connectivity
#         # Noise
#         public NetworkNoiseCy noise


cdef class NetworkDataCy:

    cdef:
        public NetworkStatesCy states
        public NetworkStatesCy derivatives
        public DoubleArray2D external_inputs
        public DoubleArray2D outputs
        public DoubleArray1D curr_outputs
        public NetworkConnectivityCy connectivity
        public NetworkNoiseCy noise
        # public NodeDataCy[:] nodes


cdef class NetworkStatesCy(DoubleArray2D):
    """ State array """

    cdef public UITYPEv1 indices
    cdef public DTYPEv1 current


cdef class NetworkConnectivityCy:
    """ Network connectivity array """

    cdef:
        public DTYPEv1 weights
        public UITYPEv1 node_indices
        public UITYPEv1 edge_indices
        public UITYPEv1 index_offsets


cdef class NetworkNoiseCy:
    """ Noise data array """

    cdef:
        public DTYPEv1 states
        public UITYPEv1 indices
        public DTYPEv1 drift
        public DTYPEv1 diffusion
        public DTYPEv1 outputs


# cdef class NodeDataCy:
#     """ Node data """
#     cdef:
#         public DoubleArray2D states
#         public DoubleArray2D derivatives
#         public DoubleArray1D output
#         public DoubleArray1D external_input
