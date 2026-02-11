""" Network Data """

from typing import Any
import numpy as np

from farms_core.array.array_cy cimport (DoubleArray1D, DoubleArray2D, IntegerArray1D)

include 'types.pxd'


class NetworkDataCy:

    DoubleArray1D times
    NetworkStatesCy states
    NetworkStatesCy derivatives
    DoubleArray1D external_inputs
    DoubleArray1D outputs
    DoubleArray1D tmp_outputs
    NetworkConnectivityCy connectivity
    NetworkNoiseCy noise

    def __init__(self):
        pass


class NetworkLogCy:
    cdef:
        public DoubleArray1D times
        public NetworkLogStatesCy states
        public DoubleArray2D external_inputs
        public DoubleArray2D outputs
        public NetworkConnectivityCy connectivity
        public NetworkNoiseCy noise


class NetworkStatesCy(DoubleArray1D):
    """ State array """
    cdef:
        public UITYPEv1 indices


class NetworkLogStatesCy(DoubleArray2D):
    """ State array for logging """
    cdef:
        public UITYPEv1 indices


class NetworkConnectivityCy:
    """ Network connectivity array """
    public DTYPEv1 weights
    public UITYPEv1 node_indices
    public UITYPEv1 edge_indices
    public UITYPEv1 index_offsets

    def __init__(
        self,
        node_indices: NDArray[(Any,), np.uintc],
        edge_indices: NDArray[(Any,), np.uintc],
        weights: NDArray[(Any,), np.double],
        index_offsets: NDArray[(Any,), np.uintc],
    ):
        super().__init__()


class NetworkNoiseCy:
    """ Noise data array """
    cdef:
        public DTYPEv1 states
        public UITYPEv1 indices
        public DTYPEv1 drift
        public DTYPEv1 diffusion
        public DTYPEv1 outputs
