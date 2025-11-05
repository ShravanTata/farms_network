from .ornstein_uhlenbeck_cy import OrnsteinUhlenbeckCy

from ..core.options import NetworkOptions


class OrnsteinUhlenbeck:
    """ OrnsteinUhlenbeck Noise Model """

    def __init__(self, network_options: NetworkOptions):
        """ Init """
        self.noise_options = [
            node.noise
            for node in network_options.nodes
            if node.noise
            if node.noise.is_stochastic
        ]

        self.n_dim = len(self.noise_options)
        self.timestep = network_options.integration.timestep
        self.seed = network_options.random_seed
        self._ou_cy = OrnsteinUhlenbeckCy(self.noise_options, self.seed)
