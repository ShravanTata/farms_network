from ornstein_uhlenbeck_cy import OrnsteinUhlenbeckCy


class OrnsteinUhlenbeck:
    """ OrnsteinUhlenbeck Noise Model """

    def __init__(
            self, timestep: float, seed: int, noise_options: List[OrnsteinUhlenbeckOptions]
    ):
        """ Init """
        assert all([opt.is_stochastic for opt in noise_options]), f"Invalid noise options{noise_options} "
        self.noise_options = noise_options
        self.n_dim = len(self.noise_options)
        self.timestep = timestep
        self.seed = seed
        self._ou_cy = O
