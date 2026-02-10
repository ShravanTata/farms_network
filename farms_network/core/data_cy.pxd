""" Network Data """


from farms_core.array.array_cy cimport (DoubleArray1D, DoubleArray2D, IntegerArray1D)


include 'types.pxd'


cdef class NetworkDataCy:
    cdef:
        public DoubleArray1D times
        public NetworkStatesCy states
        public NetworkStatesCy derivatives
        public DoubleArray1D external_inputs
        public DoubleArray1D outputs
        public DoubleArray1D tmp_outputs
        public NetworkConnectivityCy connectivity
        public NetworkNoiseCy noise


cdef class NetworkLogCy:
    cdef:
        public DoubleArray1D times
        public NetworkLogStatesCy states
        public DoubleArray2D external_inputs
        public DoubleArray2D outputs
        public NetworkConnectivityCy connectivity
        public NetworkNoiseCy noise


cdef class NetworkStatesCy(DoubleArray1D):
    """ State array """
    cdef:
        public UITYPEv1 indices


cdef class NetworkLogStatesCy(DoubleArray2D):
    """ State array for logging """
    cdef:
        public UITYPEv1 indices


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
