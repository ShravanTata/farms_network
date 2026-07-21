"""Network Data"""

import numpy as np
import numpy.typing as npt

from farms_core.array.array_cy import DoubleArray1D, DoubleArray2D


class NetworkNodeParametersCy(DoubleArray1D):
    """Node parameters array"""
    indices: npt.NDArray[np.uintc]


class NetworkEdgeParametersCy(DoubleArray1D):
    """Edge parameters array"""
    indices: npt.NDArray[np.uintc]


class NetworkStatesCy(DoubleArray1D):
    """State array"""
    indices: npt.NDArray[np.uintc]


class NetworkLogStatesCy(DoubleArray2D):
    """State array for logging"""
    indices: npt.NDArray[np.uintc]


class NetworkConnectivityCy:
    """Network connectivity array"""
    weights: npt.NDArray[np.double]
    node_indices: npt.NDArray[np.uintc]
    edge_indices: npt.NDArray[np.uintc]
    index_offsets: npt.NDArray[np.uintc]

    def __init__(
        self,
        node_indices: npt.NDArray[np.uintc],
        edge_indices: npt.NDArray[np.uintc],
        weights: npt.NDArray[np.double],
        index_offsets: npt.NDArray[np.uintc],
    ) -> None: ...


class NetworkNoiseCy:
    """Noise data array"""
    states: npt.NDArray[np.double]
    indices: npt.NDArray[np.uintc]
    drift: npt.NDArray[np.double]
    diffusion: npt.NDArray[np.double]
    outputs: npt.NDArray[np.double]


class NetworkDataCy:
    """Network data"""
    times: DoubleArray1D
    states: NetworkStatesCy
    derivatives: NetworkStatesCy
    external_inputs: DoubleArray1D
    outputs: DoubleArray1D
    tmp_outputs: DoubleArray1D
    connectivity: NetworkConnectivityCy
    noise: NetworkNoiseCy
    parameters: NetworkNodeParametersCy
    edge_parameters: NetworkEdgeParametersCy

    def __init__(self) -> None: ...


class NetworkLogCy:
    """Network log data"""
    times: DoubleArray1D
    states: NetworkLogStatesCy
    external_inputs: DoubleArray2D
    outputs: DoubleArray2D
    connectivity: NetworkConnectivityCy
    noise: NetworkNoiseCy
