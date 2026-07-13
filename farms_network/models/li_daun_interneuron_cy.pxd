""" Leaky Integrate and Fire InterNeuron Based on Daun et.al. """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EXCITATORY, INHIBITORY, CHOLINERGIC


cdef enum:
    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_H = 1
