"""

Main data structure for the network

"""

from pprint import pprint
from typing import Dict, Iterable, List, Tuple, Union, overload, Optional

import numpy as np
from farms_core import pylog
from farms_core.array.array import to_array
from farms_core.array.array_cy import (DoubleArray1D, DoubleArray2D,
                                       IntegerArray1D)
from farms_core.array.types import (NDARRAY_V1, NDARRAY_V1_D, NDARRAY_V2_D,
                                    NDARRAY_V3_D)
from farms_core.io.hdf5 import dict_to_hdf5, hdf5_to_dict

from .data_cy import (NetworkConnectivityCy, NetworkDataCy, NetworkLogCy, NetworkNoiseCy,
                      NetworkStatesCy, NetworkLogStatesCy)
from .options import NetworkOptions, NodeOptions, NodeStateOptions


NPDTYPE = np.float64
NPUITYPE = np.uintc


class NetworkStates(NetworkStatesCy):

    def __init__(self, array, indices):
        super().__init__(array, indices)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        nstates = 0
        indices = [0,]
        for index, node in enumerate(nodes):
            nstates += node._nstates
            indices.append(nstates)
        return cls(
            array=np.array(np.zeros((nstates,)), dtype=NPDTYPE),
            indices=np.array(indices)
        )

    def to_dict(self, iteration: Optional[int] = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array),
            'indices': to_array(self.indices),
        }


class NetworkConnectivity(NetworkConnectivityCy):

    def __init__(self, node_indices, edge_indices, weights, index_offsets):
        super().__init__(node_indices, edge_indices, weights, index_offsets)
        self.names: List[str] = []
        # Reverse map: original edge index → sorted position in CSR arrays
        self._orig_to_sorted = np.empty(len(edge_indices), dtype=NPUITYPE)
        for sorted_pos in range(len(edge_indices)):
            self._orig_to_sorted[edge_indices[sorted_pos]] = sorted_pos

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        edges = network_options.edges
        nedges = len(edges)
        nnodes = len(nodes)

        if nedges == 0:
            raise ValueError(
                "Network must have at least one edge. "
                "Networks with zero edges are not currently supported."
            )

        # O(1) name→index lookup instead of O(n) list.index()
        name_to_index = {node.name: i for i, node in enumerate(nodes)}

        # Build source, target, weight, edge_index arrays
        source_indices = np.empty(nedges, dtype=NPUITYPE)
        target_indices = np.empty(nedges, dtype=NPUITYPE)
        edge_weights = np.empty(nedges, dtype=NPDTYPE)
        orig_edge_indices = np.arange(nedges, dtype=NPUITYPE)

        for i, edge in enumerate(edges):
            source_indices[i] = name_to_index[edge.source]
            target_indices[i] = name_to_index[edge.target]
            edge_weights[i] = edge.weight

        # Sort by target node using numpy (C-level sort)
        sort_order = np.argsort(target_indices, kind='stable')
        source_indices = source_indices[sort_order]
        target_indices = target_indices[sort_order]
        edge_weights = edge_weights[sort_order]
        orig_edge_indices = orig_edge_indices[sort_order]

        # Build CSR index_offsets directly from sorted target indices
        index_offsets = np.zeros(nnodes + 1, dtype=NPUITYPE)
        np.add.at(index_offsets[1:], target_indices, 1)
        np.cumsum(index_offsets, out=index_offsets)

        return cls(
            node_indices=source_indices,
            edge_indices=orig_edge_indices,
            weights=edge_weights,
            index_offsets=index_offsets,
        )

    def to_dict(self, iteration: Optional[int] = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'node_indices': to_array(self.node_indices),
            'edge_indices': to_array(self.edge_indices),
            'weights': to_array(self.weights),
            'index_offsets': to_array(self.index_offsets),
        }


class NetworkNoise(NetworkNoiseCy):
    """ Data for network noise modeling """

    def __init__(self, states, indices, drift, diffusion, outputs):
        super().__init__(states, indices, drift, diffusion, outputs)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        n_noise_states = 0
        n_nodes = len(nodes)

        indices = []
        for index, node in enumerate(nodes):
            if node.noise:
                if node.noise.is_stochastic:
                    n_noise_states += 1
                    indices.append(index)

        return cls(
            states=np.full(
                shape=n_noise_states,
                fill_value=0.0,
                dtype=NPDTYPE,
            ),
            drift=np.full(
                shape=n_noise_states,
                fill_value=0.0,
                dtype=NPDTYPE,
            ),
            diffusion=np.full(
                shape=n_noise_states,
                fill_value=0.0,
                dtype=NPDTYPE,
            ),
            indices=np.array(
                indices,
                dtype=NPUITYPE,
            ),
            outputs=np.full(
                shape=n_nodes,
                fill_value=0.0,
                dtype=NPDTYPE,
            )
        )

    def to_dict(self, iteration: Optional[int] = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'states': to_array(self.states),
            'indices': to_array(self.indices),
            'drift': to_array(self.drift),
            'diffusion': to_array(self.diffusion),
            'outputs': to_array(self.outputs),
        }


class NetworkLogStates(NetworkLogStatesCy):

    def __init__(self, array, indices):
        super().__init__(array, indices)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        nstates = 0
        indices = [0,]
        buffer_size = network_options.logs.buffer_size
        for index, node in enumerate(nodes):
            nstates += node._nstates
            indices.append(nstates)
        return cls(
            array=np.array(np.zeros((buffer_size, nstates)), dtype=NPDTYPE),
            indices=np.array(indices)
        )

    def to_dict(self, iteration: Optional[int] = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array),
            'indices': to_array(self.indices),
        }


class NetworkLog(NetworkLogCy):
    """ Network Logs """

    def __init__(
        self,
        times,
        states,
        connectivity,
        outputs,
        external_inputs,
        noise,
        nodes,
        edges,
        **kwargs,
    ):
        """ Network data structure """

        super().__init__()

        self.times = times
        self.states = states
        self.connectivity = connectivity
        self.outputs = outputs
        self.external_inputs = external_inputs
        self.noise = noise

        self.nodes: Nodes = nodes
        self.edges: Edges = edges

        # assert that the data created is c-contiguous
        assert self.states.array.is_c_contig()
        assert self.outputs.array.is_c_contig()
        assert self.external_inputs.array.is_c_contig()

    @classmethod
    def from_options(cls, network_options: NetworkOptions):
        """ From options """

        buffer_size = network_options.logs.buffer_size

        times = DoubleArray1D(
            array=np.full(
                shape=buffer_size,
                fill_value=0,
                dtype=NPDTYPE,
            )
        )
        states = NetworkLogStates.from_options(network_options)

        connectivity = NetworkConnectivity.from_options(network_options)

        noise = NetworkNoise.from_options(network_options)

        outputs = DoubleArray2D(
            array=np.full(
                shape=(buffer_size, len(network_options.nodes)),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )

        external_inputs = DoubleArray2D(
            array=np.full(
                shape=(buffer_size, len(network_options.nodes)),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )

        nodes = Nodes(network_options, states, outputs, external_inputs)
        edges = Edges(network_options, connectivity=connectivity)

        return cls(
            times=times,
            states=states,
            connectivity=connectivity,
            outputs=outputs,
            external_inputs=external_inputs,
            noise=noise,
            nodes=nodes,
            edges=edges,
        )

    def to_dict(self, iteration: Optional[int] = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'times': to_array(self.times.array),
            'states': self.states.to_dict(),
            'connectivity': self.connectivity.to_dict(),
            'outputs': to_array(self.outputs.array),
            'external_inputs': to_array(self.external_inputs.array),
            'noise': self.noise.to_dict(),
            # 'nodes': {node.name: node.to_dict() for node in self.nodes},
            # 'edges': {(edge.source, edge.target): edge.to_dict() for edge in self.edges},
        }

    def to_file(self, filename: str, iteration: Optional[int] = None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to %s', filename)


class NodeStates:
    def __init__(self, network_states, node_index: int, node_name: str):
        self.node_name = node_name
        self._network_states = network_states
        self._node_index = node_index
        self.ndim = self._network_states.array.ndim
        start: int = self._network_states.indices[self._node_index]
        end: int = self._network_states.indices[self._node_index + 1]
        if start == end:
            self._has_states = False
        else:
            self._start_idx = start
            self._end_idx = end
            self._has_states = True

    @property
    def values(self):
        if not self._has_states:
            raise ValueError(f"Node {self.node_name} has no states")

        if self.ndim == 1:
            return self._network_states.array[self._start_idx:self._end_idx]
        return self._network_states.array[:, self._start_idx:self._end_idx]

    @values.setter
    def values(self, v: np.ndarray):
        if not self._has_states:
            raise ValueError(f"Node {self.node_name} has no states to be set")
        assert v.dtype == np.float_, "Values must be of type double/float"

        if self.ndim == 1:
            self._network_states.array[self._start_idx:self._end_idx] = v[:]
            return

        raise AttributeError("Cannot assign to values in logging mode.")


class NodeOutput:
    def __init__(self, network_outputs, node_index: str, node_name: str):
        self.node_name = node_name
        self._network_outputs = network_outputs
        self.ndim = self._network_outputs.array.ndim
        self._node_index = node_index

    @property
    def values(self):
        if self.ndim == 1:
            return self._network_outputs.array[self._node_index]
        return self._network_outputs.array[:, self._node_index]

    @values.setter
    def values(self, v: float):

        if self.ndim == 1:
            self._network_outputs.array[self._node_index] = v
            return
        raise AttributeError("Cannot assign to values in logging mode.")


class NodeExternalInput:
    def __init__(self, network_external_inputs, node_index: int, node_name: str):
        self.node_name = node_name
        self._network_external_inputs = network_external_inputs
        self.ndim = self._network_external_inputs.array.ndim
        self._node_index = node_index

    @property
    def values(self):
        if self.ndim == 1:
            return self._network_external_inputs.array[self._node_index]
        return self._network_external_inputs.array[:, self._node_index]

    @values.setter
    def values(self, v: float):
        if self.ndim == 1:
            self._network_external_inputs.array[self._node_index] = v
            return
        raise AttributeError("Cannot assign to values in logging mode.")


class NodeData:
    """ Accesssor for Node Data """
    def __init__(
        self,
        name: str,
        states: NodeStates,
        output: NodeOutput,
        external_input: NodeExternalInput,
    ):
        super().__init__()
        self.name: str = name
        self.states: NodeStates = states
        self.output: NodeOutput = output
        self.external_input: NodeExternalInput = external_input


class Nodes:
    """ Nodes """

    def __init__(self, network_options: NetworkOptions, states, outputs, external_inputs):
        self._nodes = []
        self._name_to_index = {}

        for idx, node_opt in enumerate(network_options.nodes):
            node = NodeData(
                node_opt.name,
                NodeStates(states, idx, node_opt.name),
                NodeOutput(outputs, idx, node_opt.name),
                NodeExternalInput(external_inputs, idx, node_opt.name),
            )
            self._nodes.append(node)
            self._name_to_index[node_opt.name] = idx

    def __getitem__(self, key: str):
        # Access by index
        if isinstance(key, int):
            return self._nodes[key]
        # Access by name
        return self._nodes[self._name_to_index[key]]

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def names(self):
        return list(self._name_to_index.keys())


class EdgeWeight:
    def __init__(self, network_connectivity, edge_index: int):
        self._network_connectivity: NetworkConnectivity = network_connectivity
        self.ndim: int = self._network_connectivity.weights.ndim
        self._edge_index: int = edge_index

    @property
    def values(self):
        if self.ndim == 1:
            return self._network_connectivity.weights[self._edge_index]
        return self._network_connectivity.weights[:, self._edge_index]

    @values.setter
    def values(self, v: float):

        if self.ndim == 1:
            self._network_connectivity.weights[self._edge_index] = v
            return
        raise AttributeError("Cannot assign to values in logging mode.")


class EdgeData:
    """ Accesssor for Edge Data """
    def __init__(
        self,
        source: str,
        target: str,
        weight: EdgeWeight
    ):
        super().__init__()
        self.source: str = source
        self.target: str = target
        self.weight: EdgeWeight = weight


class Edges:
    """ Edges """

    def __init__(self, network_options: NetworkOptions, connectivity):
        self._edges: List['EdgeData'] = []
        self._name_to_index = {}

        for idx, edge_opt in enumerate(network_options.edges):
            edge = EdgeData(
                source=edge_opt.source,
                target=edge_opt.target,
                weight=EdgeWeight(
                    network_connectivity=connectivity,
                    edge_index=connectivity._orig_to_sorted[idx],
                )
            )
            self._edges.append(edge)
            self._name_to_index[(edge_opt.source, edge_opt.target)] = idx

    @overload
    def __getitem__(self, key: int) -> EdgeData:
        pass

    @overload
    def __getitem__(self, key: Tuple[str, str]) -> EdgeData:
        pass

    def __getitem__(self, key: Union[int, tuple]) -> EdgeData:
        # Access by index
        if isinstance(key, int):
            return self._edges[key]
        # Access by name
        return self._edges[self._name_to_index[key]]

    def __len__(self):
        return len(self._edges)

    def __iter__(self):
        return iter(self._edges)

    def names(self):
        return list(self._name_to_index.keys())


class NetworkData(NetworkDataCy):
    """ Network data """

    def __init__(
        self,
        states: NetworkStates,
        derivatives: NetworkStates,
        connectivity: NetworkConnectivity,
        outputs: DoubleArray1D,
        tmp_outputs: DoubleArray1D,
        external_inputs: DoubleArray1D,
        noise: NetworkNoise,
        nodes: Nodes,
        edges: Edges,
    ):
        """ Network data structure """

        super().__init__()

        self.states: NetworkStates = states
        self.derivatives: NetworkStates = derivatives
        self.connectivity: NetworkConnectivity = connectivity
        self.outputs: DoubleArray1D = outputs
        self.tmp_outputs: DoubleArray1D = tmp_outputs
        self.external_inputs: DoubleArray1D = external_inputs
        self.noise: NetworkNoise = noise

        self.nodes: Nodes = nodes
        self.edges: Edges = edges

        # assert that the data created is c-contiguous
        assert self.states.array.is_c_contig()
        assert self.derivatives.array.is_c_contig()
        assert self.outputs.array.is_c_contig()
        assert self.tmp_outputs.array.is_c_contig()
        assert self.external_inputs.array.is_c_contig()

    @classmethod
    def from_options(cls, network_options: NetworkOptions):
        """ From options """

        states = NetworkStates.from_options(network_options)
        derivatives = NetworkStates.from_options(network_options)
        connectivity = NetworkConnectivity.from_options(network_options)

        outputs = DoubleArray1D(
            array=np.full(
                shape=(len(network_options.nodes),),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )

        tmp_outputs = DoubleArray1D(
            array=np.full(
                shape=(len(network_options.nodes),),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )

        external_inputs = DoubleArray1D(
            array=np.full(
                shape=len(network_options.nodes),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )
        nodes = Nodes(network_options, states, outputs, external_inputs)
        edges = Edges(network_options, connectivity)

        noise = NetworkNoise.from_options(network_options)

        return cls(
            states=states,
            derivatives=derivatives,
            connectivity=connectivity,
            outputs=outputs,
            tmp_outputs=tmp_outputs,
            external_inputs=external_inputs,
            noise=noise,
            nodes=nodes,
            edges=edges,
        )

    def to_dict(self, iteration: Optional[int] = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'times': to_array(self.times.array),
            'states': self.states.to_dict(),
            'derivatives': self.derivatives.to_dict(),
            'connectivity': self.connectivity.to_dict(),
            'outputs': to_array(self.outputs.array),
            'tmp_outputs': to_array(self.tmp_outputs.array),
            'external_inputs': to_array(self.external_inputs.array),
            'noise': self.noise.to_dict(),
            'nodes': {node.name: node.to_dict() for node in self.nodes},
            'edges': {(edge.source, edge.target): edge.to_dict()
                      for edge in self.edges},
        }

    def to_file(self, filename: str, iteration: Optional[int] = None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to %s', filename)
